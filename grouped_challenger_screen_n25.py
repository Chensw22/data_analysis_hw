"""Screen archived challengers over entire booking folds, carrying both unresolved release branches."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from export_budget_quarter_n23 import source
from full_roster_permutation_selection_n24 import feature_bank, ensemble_probability
from kaggle_workflow import sha256
from full_roster_conditional_pool_n23 import ROOT, data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from grouped_peer_teacher_n25 import OUT, OUTER

DEST = ROOT / 'reports/nov23_matched_learning_curve'
ART = ROOT / 'artifacts'


def verify_scored_random_composition():
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=lambda c: c != 'Name')
    random = 'proposal_n24_random_subspace'
    nine = 'proposal_n24_selected_dissent'
    parent = 'proposal_v131_party_consensus'
    native = source('proposal_online_first_counts15',test)[0]
    prev, best, old = [source(k,test)[1] for k in (parent,nine,random)]
    model_path = ART/random/'ensemble.joblib'
    spec = json.loads((ROOT/'reports'/random/'manifest.json').read_text())
    assert sha256(model_path) == spec['model_sha256']
    model = joblib.load(model_path)
    x, fields, _ = feature_bank(test)
    query = pd.DataFrame(x,columns=fields).reindex(columns=model['fields'],fill_value=0).to_numpy()
    mixed = .75*native + .25*ensemble_probability(model['models'],query)
    np.testing.assert_array_equal(np.where((native>=.5)==prev,mixed>=.5,prev),old)
    actual = np.where(((native>=.5)==prev)&(prev==best),mixed>=.5,best)
    changes, repairs = old != prev, best != prev
    assert not (changes & repairs).any()
    np.testing.assert_array_equal(actual != prev,changes | repairs)
    def public(name):
        receipt, = (ROOT/'reports'/name/'online_attempts').glob('*.json')
        return float(json.loads(receipt.read_text())['evaluation']['Public Score'])
    return dict(random_model=random, calibrated_weight=.25, formula='score(random) + score(nine) - score(parent)',
                old_random_score=public(random), old_nine_score=public(nine), old_parent_score=public(parent),
                disjoint_random_edits=int(changes.sum()), disjoint_nine_edits=int(repairs.sum()),
                overlap=int((changes & repairs).sum()), differences_from_incumbent=int((actual!=best).sum()),
                implied_score_from_rounded_receipts=round(public(random)+public(nine)-public(parent),5))


def run():
    raw, y, groups, full_fold = data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows, hf, hn, hv, hs = [z[k] for k in ('rows', 'fold', 'nine', 'v131', 'scores')]
    native_h = hs[:, 4]
    hfree = (hn == hv) & (hv == (native_h >= .5))
    alternatives, panels = prediction_sets(raw, y, groups, full_fold, rows, hf)
    names = (['response_channels/'+k for k in ('additive','any_event','odd_events')] +
             ['broad_error_transport_projection/'+k for k in ('linear','fixed_curves','learned_curves')] + alternatives)
    full, old = [np.column_stack(v) for v in panels]
    assert len(set(names)) == old.shape[1] == full.shape[1] == 68
    packets = []
    for f in OUTER:
        with np.load(OUT/f'release{f}.npz') as z:
            np.testing.assert_array_equal(z['q'], np.flatnonzero(full_fold == f))
            np.testing.assert_array_equal(z['y'], y[z['q']])
            packets.append({k:z[k] for k in z.files})
    joined = {k:np.concatenate([z[k] for z in packets]) for k in packets[0]}
    q, p, v131, nine, party = [joined[k] for k in ('q','native','v131','nine','party')]
    assert len(np.unique(q)) == len(q)
    low, high = joined['nine_on_low_branch'], joined['nine_on_high_branch']
    resolved = (v131 >= 0) & (nine >= 0)
    common = resolved & ((full[q] >= .5) == nine[:, None]).all(1)
    new_orders = ~np.isin(groups[q], groups[rows])
    records = []
    for j, name in enumerate(names):
        for w in (.25, 1.):
            pred = ((1-w)*native_h + w*old[:, j]) >= .5
            hdiff = np.where(hfree & (hn != pred), np.where(hn == y[rows], -1, 1), 0)
            pred = (1-w)*p + w*full[q, j] >= .5
            lo, hi = edit_effect_bounds(pred, p >= .5, v131, low, high, y[q])
            assert (lo[resolved] == hi[resolved]).all()
            record = dict(model=name, weight=w, historical_net=int(hdiff.sum()),
                          historical_min_fold=min(int(hdiff[hf==k].sum()) for k in np.unique(hf)),
                          known_fixed=int((lo[resolved]>0).sum()), known_harmed=int((lo[resolved]<0).sum()),
                          nine_common_fixed=int((lo[common]>0).sum()), nine_common_harmed=int((lo[common]<0).sum()),
                          net_low=int(lo.sum()), net_high=int(hi.sum()), party_low=int(lo[party].sum()),
                          party_high=int(hi[party].sum()), new_order_low=int(lo[new_orders].sum()),
                          new_order_high=int(hi[new_orders].sum()))
            for f in OUTER:
                use = full_fold[q] == f
                record[f'low_{f}'] = int(lo[use].sum())
                record[f'high_{f}'] = int(hi[use].sum())
            records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(DEST/'grouped_challenger_screen.csv', index=False)
    best_old = table.sort_values(['historical_min_fold','historical_net','model','weight'],
                                 ascending=[False,False,True,True]).iloc[0]
    robust = table[(table.net_low > 0) & (table.historical_net > 0)]
    both = robust[(robust.historical_min_fold >= 0) & (robust[[f'low_{f}' for f in OUTER]].min(axis=1) >= 0)]
    report = dict(folds=list(OUTER), passengers=len(q), parties=int(party.sum()), new_order_passengers=int(new_orders.sum()),
                  resolved=int(resolved.sum()), original_consensus_errors=int((common & (nine != y[q])).sum()),
                  original_consensus_correct=int((common & (nine == y[q])).sum()), choices=len(table),
                  historical_choice=json.loads(best_old.to_json()),
                  positive_historical_and_new_lower_bound=json.loads(robust.to_json(orient='records')),
                  nonnegative_each_fold=json.loads(both.to_json(orient='records')))
    # Replicate preselected rules on the other five folds. Nine-vote evidence here
    # is limited to the original structural certificate; native comparisons cover all rows.
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['fold'], full_fold)
        np.testing.assert_array_equal(z['y'], y)
        rp, rn, rc = z['v96'], z['certified_nine'], z['certified_v131_equals_v96']
    fresh = ~np.isin(groups, groups[rows])
    holdout = ~np.isin(full_fold, OUTER)
    selected = {'historical_only':best_old}
    if len(robust):
        selected['historical_plus_three_folds'] = robust.sort_values(
            ['net_low','historical_min_fold','model','weight'], ascending=[False,False,True,True]).iloc[0]
    report['remaining_five_folds'] = {}
    for source, model in selected.items():
        j, w = names.index(model['model']), float(model['weight'])
        pred = (1-w)*rp + w*full[:,j] >= .5
        native_hard = rp >= .5
        exact = np.where(rn == native_hard, pred, rn)
        checks = {}
        for label, mask, before, after in [
                ('all_vs_native',holdout, native_hard,pred), ('certified_vs_nine',holdout & rc,rn,exact),
                ('new_order_vs_native',holdout & fresh,native_hard,pred),
                ('new_order_certified_vs_nine',holdout & fresh & rc,rn,exact)]:
            d = (after[mask] == y[mask]).astype(int) - (before[mask] == y[mask])
            checks[label] = dict(rows=int(mask.sum()),fixed=int((d>0).sum()),harmed=int((d<0).sum()),
                                 interval_pp=interval(d,groups[mask]),
                                 folds=[int(d[full_fold[mask]==k].sum()) for k in np.unique(full_fold[mask])])
        report['remaining_five_folds'][source] = dict(model=model['model'], weight=w, checks=checks)
    if any(item['model'] == 'permutation_selection/random' and item['weight'] == .25
           for item in report['remaining_five_folds'].values()):
        report['scored_composition_check'] = verify_scored_random_composition()
    (DEST/'grouped_challenger_screen.json').write_text(json.dumps(report, indent=2)+'\n')
    print('selected from historical data', best_old.to_dict(), flush=True)
    print('positive on both', len(robust), 'stable',len(both), flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
