"""Aggregate data diagnostics; only frozen development target rows are parsed.

Run: .venv-mitra/bin/python data_quality_diagnostic.py
No model fits, OOF reads, target cleaning, or per-passenger output.
"""
from pathlib import Path
import hashlib
import itertools
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp

from audit_top5_references import DEV_HASH, read_development_only, row_hash

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/data_error_diagnosis_20261117/data_quality.json'
SPEND = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
NUMERIC = ['Age', *SPEND, 'CabinNumber']
CATEGORICAL = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
COMPACT = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', *SPEND,
           'Deck', 'CabinNumber', 'Side']
SOURCES = ['audit_top5_references.py', 'rule_ablation.py', 'mitra_candidate.py',
           'reports/attribute_mask_research.json', 'reports/relational_review/diagnostics.json',
           'reports/top5_missing_flags_review.json',
           'reports/remaining_method_coverage_review_20261117.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attributes(frame):
    assert 'Transported' not in frame
    x = frame.copy()
    parts = x.Cabin.str.split('/', expand=True)
    x['Deck'], x['Side'] = parts[0], parts[2]
    x['CabinNumber'] = pd.to_numeric(parts[1], errors='coerce')
    x['GroupId'] = x.PassengerId.str.split('_').str[0]
    x['GroupNumber'] = pd.to_numeric(x.GroupId, errors='raise')
    return x


def rate(mask):
    mask = np.asarray(mask, dtype=bool)
    return {'rows': int(mask.sum()), 'fraction': float(mask.mean())}


def target_summary(mask, y):
    selected = np.asarray(y)[np.asarray(mask, dtype=bool)]
    return {'rows': len(selected), 'positive_labels': int(selected.sum()),
            'positive_fraction': float(selected.mean()) if len(selected) else None}


def frame_hash(frame):
    # Attributes only. Avoid hashing/parsing the unused target values in raw CSV.
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def duplicate_summary(frame, y, cols):
    # Internal keys are never exported; missing is a distinct, explicit state.
    keys = frame[cols].astype('string').fillna('<NA>')
    t = keys.copy(); t['_y'] = y
    g = t.groupby(cols, dropna=False, sort=False)['_y'].agg(['size', 'sum'])
    repeated = g['size'].gt(1)
    mixed = g['sum'].gt(0) & g['sum'].lt(g['size'])
    return {'columns': cols, 'distinct_patterns': len(g),
            'repeated_patterns': int(repeated.sum()), 'rows_in_repeated_patterns': int(g.loc[repeated, 'size'].sum()),
            'mixed_label_patterns': int(mixed.sum()), 'rows_in_mixed_label_patterns': int(g.loc[mixed, 'size'].sum()),
            'empirical_minority_labels_in_exact_patterns': int(np.minimum(g['sum'], g['size']-g['sum']).sum()),
            'interpretation': 'Descriptive ambiguity for this chosen representation on reused dev labels, not verified mislabels, a population Bayes bound, or an 80% accuracy ceiling.'}


def missingness(dev, y, train, test):
    per_column = {}
    for c in COMPACT:
        miss = dev[c].isna()
        strata = []
        for (planet, cryo), part in dev.groupby(['HomePlanet', 'CryoSleep'], dropna=False, sort=False):
            idx = part.index.to_numpy()
            strata.append({'HomePlanet': None if pd.isna(planet) else str(planet),
                           'CryoSleep': None if pd.isna(cryo) else bool(cryo),
                           'missing': target_summary(part[c].isna(), y[idx]),
                           'observed': target_summary(part[c].notna(), y[idx])})
        per_column[c] = {'development': rate(miss), 'full_train_attributes': rate(train[c].isna()),
                         'test_attributes': rate(test[c].isna()),
                         'development_labels_missing': target_summary(miss, y),
                         'development_labels_observed': target_summary(~miss, y),
                         'planet_cryo_strata': strata}
    counts = dev[COMPACT].isna().sum(axis=1)
    patterns = dev[COMPACT].isna().astype(int).astype(str).agg(''.join, axis=1).value_counts()
    pairs = []
    for a, b in itertools.combinations(COMPACT, 2):
        ma, mb = dev[a].isna().astype(int), dev[b].isna().astype(int)
        phi = float(ma.corr(mb)) if ma.nunique() > 1 and mb.nunique() > 1 else None
        pairs.append({'columns': [a, b], 'both_missing': int((ma.eq(1) & mb.eq(1)).sum()), 'phi': phi})
    return {'columns_in_mask_order': COMPACT, 'per_column': per_column,
            'missing_count_distribution': {str(k): int(v) for k,v in counts.value_counts().sort_index().items()},
            'missing_pattern_counts': patterns.to_dict(),
            'largest_absolute_phi_pairs_descriptive_only': sorted(pairs, key=lambda p: abs(p['phi'] or 0), reverse=True)[:10],
            'complete_vs_missing_label_rates': {'complete': target_summary(counts.eq(0), y), 'any_missing': target_summary(counts.gt(0), y)}}


def shift(train, test):
    result = {'numeric': {}, 'categorical': {}}
    for c in NUMERIC:
        a, b = train[c].dropna().to_numpy(float), test[c].dropna().to_numpy(float)
        result['numeric'][c] = {'ks_statistic_observed_only': float(ks_2samp(a, b).statistic),
                               'train_quantiles_0_25_50_75_95_100': np.quantile(a, [0,.25,.5,.75,.95,1]).tolist(),
                               'test_quantiles_0_25_50_75_95_100': np.quantile(b, [0,.25,.5,.75,.95,1]).tolist(),
                               'missing_rate_difference_test_minus_train': float(test[c].isna().mean()-train[c].isna().mean())}
    for c in CATEGORICAL:
        a, b = train[c].astype('string').fillna('<NA>'), test[c].astype('string').fillna('<NA>')
        pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        universe = pa.index.union(pb.index)
        result['categorical'][c] = {'total_variation_including_missing': float((pa.reindex(universe, fill_value=0)-pb.reindex(universe, fill_value=0)).abs().sum()/2),
                                   'test_rows_with_train_unseen_category': int((~b.isin(set(a))).sum()),
                                   'missing_rate_difference_test_minus_train': float(test[c].isna().mean()-train[c].isna().mean())}
    result['limitation'] = 'Marginal feature checks do not establish joint/conditional IID or test label balance; no importance weights inferred.'
    return result


def rules(frame):
    conditions = {'CryoSleep_true': frame.CryoSleep.eq(True), 'Age_under_13': frame.Age.lt(13)}
    result = {}
    for name, condition in conditions.items():
        sub = frame.loc[condition, SPEND]
        result[name] = {'condition_rows': int(condition.sum()), 'rows_any_observed_positive_spend': int(sub.gt(0).any(axis=1).sum()),
                        'rows_with_spend_missing': int(sub.isna().any(axis=1).sum()),
                        'missing_spend_cells': int(sub.isna().sum().sum()),
                        'per_service': {c: {'known': int(sub[c].notna().sum()), 'known_nonzero': int(sub[c].fillna(0).ne(0).sum()),
                                             'missing': int(sub[c].isna().sum())} for c in SPEND}}
    result['missing_CryoSleep_with_positive_spend'] = int((frame.CryoSleep.isna() & frame[SPEND].gt(0).any(axis=1)).sum())
    result['known_awake_with_all_five_observed_zero'] = int((frame.CryoSleep.eq(False) & frame[SPEND].notna().all(axis=1) & frame[SPEND].eq(0).all(axis=1)).sum())
    result['missing_CryoSleep_and_all_five_observed_zero'] = int((frame.CryoSleep.isna() & frame[SPEND].notna().all(axis=1) & frame[SPEND].eq(0).all(axis=1)).sum())
    result['numeric_negative_counts'] = {c: int(frame[c].lt(0).sum()) for c in NUMERIC}
    return result


def unanimous_reconstruction(roster):
    # Leave-one-observed-attribute-out diagnostic, not target prediction.
    out = {}
    for c in ['HomePlanet', 'Destination', 'Deck', 'Side', 'CryoSleep']:
        evidence = {k: {'eligible_observed': 0, 'wrong_observed': 0, 'fillable_missing': 0} for k in ['one_known_peer', 'two_or_more_known_peers']}
        mixed_groups = 0
        for _, group in roster.groupby('GroupId', sort=False):
            vals = group[c].astype('string')
            mixed_groups += int(vals.dropna().nunique() > 1)
            for index, value in vals.items():
                peer = vals.drop(index).dropna()
                if len(peer) and peer.nunique() == 1:
                    slot = evidence['one_known_peer' if len(peer)==1 else 'two_or_more_known_peers']
                    if pd.isna(value): slot['fillable_missing'] += 1
                    else:
                        slot['eligible_observed'] += 1
                        slot['wrong_observed'] += int(value != peer.iloc[0])
        for slot in evidence.values():
            slot['observed_attribute_error_rate'] = slot['wrong_observed']/slot['eligible_observed'] if slot['eligible_observed'] else None
        out[c] = {'groups_with_multiple_known_values': mixed_groups, 'leave_one_observed_value_out': evidence}
    return out


def coverage(train, test, dev):
    result = {}
    for c in ['PassengerId', 'GroupId', 'Cabin']:
        a,b = set(train[c].dropna()),set(test[c].dropna())
        result[c] = {'train_unique_known': len(a), 'test_unique_known': len(b), 'shared_known_keys': len(a & b),
                     'test_rows_with_training_attribute_key': int(test[c].notna().where(test[c].isin(a), False).sum())}
    for name, frame in [('train', train), ('test', test), ('development', dev)]:
        sizes=frame.groupby('GroupId').size()
        result[name] = {'rows': len(frame), 'duplicate_passenger_ids': int(frame.PassengerId.duplicated().sum()),
                        'malformed_passenger_ids': int((~frame.PassengerId.str.fullmatch(r'\d{4}_\d{2}')).sum()),
                        'known_cabins_crossing_group_ids': int(frame.dropna(subset=['Cabin']).groupby('Cabin').GroupId.nunique().gt(1).sum()),
                        'group_size_histogram': {str(k):int(v) for k,v in sizes.value_counts().sort_index().items()},
                        'rows_in_multi_member_groups': int(frame.GroupId.map(sizes).gt(1).sum())}
    fullsize=train.groupby('GroupId').size(); devsize=dev.groupby('GroupId').size()
    result['development_graph_missing_roster_members'] = {'groups': int(devsize.lt(fullsize.reindex(devsize.index)).sum()),
          'dev_rows': int((dev.GroupId.map(devsize)<dev.GroupId.map(fullsize)).sum())}
    return result


def confounding_and_space(dev, y, roster, dev_ids, ntrain):
    """Fixed descriptive strata; no cutpoint search and no model predictions."""
    d = dev.copy(); d['_y'] = y
    d['AgeBand'] = pd.cut(d.Age, [-np.inf, 12, 30, np.inf], labels=['0_12', '13_30', '31_plus']).astype('string').fillna('missing')
    d['SpendState'] = np.where(d[SPEND].gt(0).any(axis=1), 'any_observed_positive',
                              np.where(d[SPEND].notna().all(axis=1), 'five_observed_zero', 'zero_or_unknown'))
    tables = {}
    for cols in [['HomePlanet','Deck'], ['HomePlanet','Deck','CryoSleep'],
                 ['HomePlanet','Deck','AgeBand'], ['HomePlanet','CryoSleep','SpendState']]:
        rows=[]
        for key, part in d.groupby(cols, dropna=False, sort=False):
            key = key if isinstance(key,tuple) else (key,)
            rows.append({**{c:None if pd.isna(v) else str(v) for c,v in zip(cols,key)},
                         'rows':len(part),'positive_labels':int(part['_y'].sum()),
                         'positive_fraction':float(part['_y'].mean()), 'small_cell_under_30':len(part)<30})
        tables['__'.join(cols)] = rows
    # Attribute-only rank, separately within Deck/Side, before looking at labels.
    rank = roster.groupby(['Deck','Side'], dropna=False).CabinNumber.rank(method='average',pct=True)
    quartile = np.minimum(np.floor(rank.fillna(0).to_numpy()*4).astype(int),3)
    quartile[rank.isna().to_numpy()] = -1
    d['SpatialQuartile'] = quartile[dev_ids]
    spatial=[]
    for key, part in d.groupby(['HomePlanet','Deck','Side','CryoSleep','SpatialQuartile'], dropna=False,sort=False):
        spatial.append({**{c:None if pd.isna(v) else str(v) for c,v in zip(['HomePlanet','Deck','Side','CryoSleep','SpatialQuartile'],key)},
                        'rows':len(part),'positive_labels':int(part['_y'].sum()),
                        'positive_fraction':float(part['_y'].mean()),'small_cell_under_30':len(part)<30})
    drift=[]
    for (deck,side), group in roster.groupby(['Deck','Side'],dropna=False,sort=False):
        idx=group.index.to_numpy(); a=quartile[idx[idx<ntrain]]; b=quartile[idx[idx>=ntrain]]
        if len(a) and len(b):
            pa=np.array([(a==q).mean() for q in [-1,0,1,2,3]])
            pb=np.array([(b==q).mean() for q in [-1,0,1,2,3]])
            drift.append({'Deck':None if pd.isna(deck) else str(deck),'Side':None if pd.isna(side) else str(side),
                          'train_rows':len(a),'test_rows':len(b),'rank_quartile_TV':float(abs(pa-pb).sum()/2)})
    earth=d.HomePlanet.eq('Earth'); g=d.Deck.eq('G')
    return {'fixed_cross_tables':tables,'Earth_DeckG_overlap':{'earth_rows':int(earth.sum()),'deckG_rows':int(g.sum()),
               'intersection_rows':int((earth&g).sum()),'earth_fraction_of_deckG':float(earth[g].mean()) if g.any() else None},
            'spatial_rank_definition':'Within full unlabeled roster Deck×Side, CabinNumber average percentile rank; four fixed equal-width rank bins, missing=-1. No Transported used to choose bins.',
            'spatial_label_tables':spatial,'train_test_spatial_rank_drift':drift,
            'limitations':'These are marginal/conditional observed target fractions, not model accuracy, residual effects, or causal deck/age effects. Many dependent small cells; no threshold/feature selected from this table.'}


def fixed_repair_coverage(roster, dev_ids, ntrain):
    """No labels: quantify one fixed package, retaining independent original frame."""
    original=roster[COMPACT].copy(); repaired=original.copy(); masks={}
    zero=roster.CryoSleep.eq(True)
    for c in SPEND:
        mask=zero & roster[c].isna()
        repaired.loc[mask,c]=0
    masks['CryoTrue_missing_spend']=zero & roster[SPEND].isna().any(axis=1)
    for c in ['HomePlanet','Side']:
        grouped=roster[c].groupby(roster.GroupId)
        value=roster.GroupId.map(grouped.first().where(grouped.nunique().eq(1)))
        masks[c]=roster[c].isna() & value.notna()
        repaired.loc[masks[c],c]=value.loc[masks[c]]
    changed=~(original.eq(repaired)|(original.isna()&repaired.isna()))
    assert not (changed & original.notna()).any().any()
    assert original.columns.equals(repaired.columns)
    masks['any_repair']=changed.any(axis=1)
    out={}
    for name,idx in [('development',dev_ids),('full_train_attributes',np.arange(ntrain)),('test_attributes',np.arange(ntrain,len(roster)))]:
        out[name]={rule:rate(mask.iloc[idx]) for rule,mask in masks.items()}
        out[name]['changed_cells']=int(changed.iloc[idx].sum().sum())
    return {'coverage':out,'observed_cells_unchanged':True,'column_order_and_count_unchanged':True,
            'only_attributes_used':True,'no_repaired_frame_exported':True}


def hypotheses(report):
    # Proposed falsifiable checks, not new candidate scores or inferred mislabels.
    return [
      {'id':'H1_conditional_zero_imputation',
       'hypothesis':'For a mean-imputing model, observed CryoSleep=True identifies missing service amounts as structural zeros; replacing only these missing amounts may retain information that unconditional means blur.',
       'evidence':{'dev':report['physical_attribute_checks']['development']['CryoSleep_true'],
                   'test_unlabeled':report['physical_attribute_checks']['test']['CryoSleep_true']},
       'already_tested':'run009 v26/v28 tested broader logical filling on CatBoost engineered inputs; run043 added group attributes/fills, run052 added Mitra missing flags. These are not the exact raw-Mitra, Cryo-only amount-fill ablation; previous failures lower the prior.',
       'minimal_test_if_authorized':'Freeze raw Mitra13 vs same checkpoint/preprocessing with only missing spend under observed CryoSleep=True set to zero, before fold-local means. No children/Cryo inference/extra flags/routing/threshold changes. Original three development folds; matched cached raw after identity checks. Verify unchanged observed cells and rules first.',
       'falsifier':'Any observed nonzero amount in the condition challenges an exact rule; if authorized CV gives no predeclared aggregate accuracy/logloss gain, archive rather than adding more rules.',
       'scope':'Representation hypothesis, not label correction; no experiment executed here.'},
      {'id':'H2_peer_unanimity_is_attribute_specific',
       'hypothesis':'A generic unanimous-peer fill can corrupt Destination/Deck/Cryo, while HomePlanet/Side may obey stronger observed group consistency; group-context bundle failures cannot isolate which assumption failed.',
       'evidence_key':'unanimous_attribute_reconstruction',
       'already_tested':'run009 unanimous fill and run043 Mitra context20 were executed bundles; surname HomePlanet run017 and soft HomePlanet marginal run029 also tested. No justification to blindly repeat these bundles.',
       'minimal_test_if_authorized':'First predeclare attribute-only group-held-out masking validation stratified by known-peer count, with no Transported usage. Only a rule with negligible held-out attribute errors and material actual-missing coverage can motivate a later matched task experiment.',
       'falsifier':'Nontrivial attribute error on observed peer-unanimous rows or tiny true-missing coverage rejects deterministic filling. Random observed-mask success still need not transfer to actual missingness.',
       'scope':'Current leave-one-value-out evidence is descriptive and transductive, not independent validation.'},
      {'id':'H3_Earth_G_confounding_and_unobserved_information',
       'hypothesis':'Earth, DeckG, age and sleep/spending composition overlap, so separate weak slice scores may describe the same hard population. Conditional spatial variation could remain, but mixed labels do not establish erroneous labels.',
       'evidence_keys':['confounding_and_space','exact_feature_label_conflicts'],
       'already_tested':'Graph/peer residual diagnostics already weak with confidence intervals crossing zero; no test/train GroupId or Cabin label-neighbor coverage. Label smoothing was tested without accuracy improvement.',
       'minimal_test_if_authorized':'First compare existing group-OOF residuals in the fixed Planet×Deck×Cryo and rank-quartile cells, using group clusters and no fitted correction. Main agent owns OOF analysis. Only repeated conditional residual structure beyond existing spatial expert can motivate a predeclared nested spatial model; no cutpoint/weight search. Separately stratify exact13/coarsened conflicts without relabeling.',
       'falsifier':'If Earth/G differences vanish after sleep/spend/age composition, or conditional spatial residuals are inconsistent/uncertain across frozen splits, do not add another spatial expert. No source-confirmed wrong labels means no label cleaning.',
       'scope':'No new robust-loss model recommended from these counts alone.'}
    ]


def main():
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as split:
        dev_ids=np.sort(split['development']).astype(np.int64)
    assert len(dev_ids)==6954 and row_hash(dev_ids)==DEV_HASH
    train_raw,test_raw,y=read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev_ids)
    train,test=attributes(train_raw),attributes(test_raw)
    dev=train.iloc[dev_ids].reset_index(drop=True)
    roster=pd.concat([train,test],ignore_index=True)
    assert 'Transported' not in roster and len(y)==6954
    report={'status':'descriptive_data_diagnostic_no_model_or_oof_scores','label_policy':'Only frozen6954dev targets parsed via audit.read_development_only; unused1739targets never parsed; test attributes only.',
      'rows':{'development':len(dev),'train_attributes':len(train),'test_attributes':len(test)},
      'identity':{'development_ids_sha256':row_hash(dev_ids),'development_labels_sha256':hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest(),
                  'train_attribute_hash':frame_hash(train_raw),'test_attribute_hash':frame_hash(test_raw)},
      'development_label_balance':target_summary(np.ones(len(y),dtype=bool),y),
      'missingness':missingness(dev,y,train,test),
      'train_test_marginal_shift':shift(train,test),
      'physical_attribute_checks':{k:rules(f) for k,f in [('development',dev),('train_attributes',train),('test',test)]},
      'exact_feature_label_conflicts':{'compact13':duplicate_summary(dev,y,COMPACT),
         'compact_without_cabin_number':duplicate_summary(dev,y,[c for c in COMPACT if c!='CabinNumber'])},
      'unanimous_attribute_reconstruction':unanimous_reconstruction(roster),
      'id_group_cabin_coverage':coverage(train,test,dev),
      'confounding_and_space':confounding_and_space(dev,y,roster,dev_ids,len(train)),
      'fixed_high_consistency_repair':fixed_repair_coverage(roster,dev_ids,len(train)),
      'existing_relation_diagnostic_reference':{'path':'reports/relational_review/diagnostics.json','sha256':sha(ROOT/'reports/relational_review/diagnostics.json'),
         'not_recomputed':'No residual graph, surname label encoding, or model-error overlap matrix computed in this task.'},
      'limitations':['Reused adaptive development labels; descriptive associations, not causal effects or independent hypothesis tests.',
                    'No individual records or label values exported; only aggregates and provenance hashes.',
                    'Identical feature vectors with different labels are not established data errors or a generalization upper bound.',
                    'Known-attribute consistency cannot prove missing-value truth; hidden values are not observable.',
                    'Univariate train/test similarity does not rule out multivariate/conditional shift.',
                    'No model fitting, OOF access, GPU, target modification, or online data used.'],
      'source_sha256':{p:sha(ROOT/p) for p in ['data_quality_diagnostic.py',*SOURCES]},
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}}
    report['testable_hypotheses']=hypotheses(report)
    report['single_fixed_experiment_judgment']={
      'decision':'Worth one precisely matched raw-Mitra13 repair-package ablation, not a claim that cleaning explains most errors. No training authorized or executed by this diagnostic.',
      'fixed_rules':['Observed CryoSleep=True: fill only missing five-service amounts with zero.',
                     'Missing HomePlanet: fill only when all known same-GroupId roster values agree.',
                     'Missing Side: same unanimous observed-group rule.'],
      'excluded_rules':['Do not fill CryoSleep, Destination, Deck, Age or CabinNumber; no child-age rule or observed-value replacement.',
                        'No new flags/counts/IDs, row filtering, label editing, or missing-only probability routing.'],
      'why_not_equivalent_to_prior_tests':'run009 logical/unanimous CatBoost used broader rules and engineered inputs; run043 Mitra context20 also added seven columns and filled Destination/Deck, which show attribute-level counterexamples. run052 flags expose missingness rather than reconstruct values. None isolates this fixed three-rule raw-Mitra13 package.',
      'fair_comparison':'Original raw13 control independently built. Candidate retains same named columns/order, categorical handling, original mirror RNG policy, checkpoint, full support, query batching and threshold. Fit preprocessing only on each training fold; numerical means and model internal quantiles may legitimately change after repairs, so do not claim all transformed observed rows remain bitwise identical.',
      'unlabeled_roster_boundary':'Group aggregates built once from full train/test attributes with Transported excluded; IDs align rows, not input features. GroupCV remains transductive attribute use, never cross-group target propagation.',
      'evaluation_proposal':'Before scores, freeze original three random folds with verified raw-Mitra cached control, full shape/resource and reload validation; report paired corrections/regressions, logloss and per-fold results. A concrete resource-continue heuristic is pooled gain>=0.001, logloss not worse, >=2 folds accuracy not worse; root must freeze it before fitting. Passing does not establish stable superiority or automatically authorize upload.',
      'stop':'If package fails its predeclared criterion, archive; do not select individual rules, reroute repaired rows, append age rules or change weights on these OOF results.',
      'counterevidence':'Observed consistency does not identify missing ground truth. Missing/complete aggregate error rates are similar, related bundles have failed, and changes are confined initially to a minority of rows. Most common model errors may reflect unobserved information or conditional structure; no claim of a one-percentage-point benefit.'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT.relative_to(ROOT)),'sha256':sha(OUT),'rows':report['rows'],
                      'compact13_conflicts':report['exact_feature_label_conflicts']['compact13'],
                      'cryo_zero_evidence':report['physical_attribute_checks']['development']['CryoSleep_true']},indent=2))


if __name__=='__main__':
    main()
