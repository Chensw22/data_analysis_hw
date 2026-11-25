"""Check whether outside experts can correct domain-appropriate consensus errors."""
import json
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from held_expert_portfolios_n26 import NEW, ROOT, FEES, panel_predictions, outcome_scores, sources
from outside_panel_coverage_n26 import projection, archives, count
from within_fold_outside_evidence_n26 import fit_probability


def calibrated(base, outside, dry, y, groups, fold, query, fresh):
    predictions = np.full((len(y), 3), np.nan)
    joined = np.column_stack([base,outside])
    for k in np.unique(fold):
        bank = np.flatnonzero(fold == k)
        if fresh:
            pieces = [(bank[~np.isin(groups[bank], groups[query])], bank[query[bank]])]
        else:
            inner = GroupKFold(3, shuffle=True, random_state=4567)
            pieces = [(bank[t], bank[q][query[bank[q]]]) for t,q in inner.split(bank, groups=groups[bank])]
        for t,q in pieces:
            if not len(q):
                continue
            assert not np.isin(groups[t],groups[q]).any()
            for j,x in enumerate([base, joined, joined]):
                train = t[dry[t]] if j==2 else t
                predictions[q,j] = fit_probability(x,train,q,y)
    assert np.isfinite(predictions[query]).all()
    return predictions


def run():
    raw,y,groups,fold,views,votes = projection(include_full=True)
    dry = ~raw[FEES].gt(0).any(axis=1).to_numpy()
    ext,_ = archives(raw,y,groups,fold,True)
    lib = ROOT/'reports/nov23_matched_learning_curve'
    groups_of_models = json.loads((lib/'outside_panel_coverage.json').read_text())['family_votes']
    fam = {n:np.mean([ext[c] for c in cols],axis=0) for n,cols in groups_of_models.items()}
    # EXAONE has a different outer fold. Use it only for fixed decisions requiring no calibration.
    direct = ['Mitra','TabDPT','LimiX2','EXAONE']
    generative = ['mask_net','bill_flow','profiles']
    safe = [k for k in fam if k!='EXAONE']
    assert NEW[-2:] == ['register_paid','sequence_paid']
    v = sources()[-1]['full']
    prob = outcome_scores(v,panel_predictions(v,'full'))
    np.testing.assert_array_equal(votes,prob>=.5)
    # The same operative panel is used for the level and dispersion of prior confidence.
    quantiles = np.quantile(prob, [.1,.5,.9],axis=1).T
    quantiles[dry] = np.quantile(prob[dry,:-2], [.1,.5,.9],axis=1).T
    stable = votes.mean(1)
    stable[dry] = votes[dry,:-2].mean(1)
    base = np.column_stack([logit(quantiles.clip(.003,.997)), stable, dry, raw.Cabin.str.startswith('G/',na=False)])
    odds = np.column_stack([logit(fam[k].clip(.003,.997)) for k in safe])
    mean_raw = np.median(base[:,:3],axis=1)[:,None]
    other = np.column_stack([odds,np.abs(odds-mean_raw)])
    report = {}
    for tag in ('full','bookingmix','historical_to_new'):
        scope,parent = views[tag]
        wrong = parent!=y
        plain = (parent[:,None]==votes).all(1)
        operative = np.where(dry,(parent[:,None]==votes[:,:-2]).all(1),plain)
        new = scope & operative & ~plain
        old = scope & plain
        contested = scope & ~operative
        assert int((new&wrong).sum())=={'full':56,'bookingmix':39,'historical_to_new':5}[tag]
        actions = {n:((v>=.5)!=parent) for n,v in ext.items()}
        fvote = {n:((v>=.5)!=parent) for n,v in fam.items()}
        direct_votes = np.column_stack([fvote[k] for k in direct]).sum(1)
        gen_votes = np.column_stack([fvote[k] for k in generative]).sum(1)
        actions.update(any_outside=np.column_stack(list(actions.values())).any(1),
                       both_types=(direct_votes>=1)&(gen_votes>=1),
                       majority_of_each=(direct_votes>=3)&(gen_votes>=2))
        pred = calibrated(base,other,dry,y,groups,fold,scope,tag=='historical_to_new')
        core = (pred[:,0]>=.5)!=parent
        pooled = (pred[:,1]>=.5)!=parent
        special = dry&((pred[:,2]>=.5)!=parent)
        actions.update(calibrated_prior=core, calibrated_outside=pooled,
                       outside_additional=pooled&~core,
                       dry_expert=special,confirmed_increment=pooled&special&~core)
        subsets = {'all':scope,'new_common':new,'old_common':old,'remaining':contested,'dry':scope&dry}
        r = {s:{n:count(m,wrong,flip) for n,flip in actions.items()} for s,m in subsets.items()}
        rows = np.flatnonzero(new)
        risk = np.where(parent[rows,None],1-pred[rows],pred[rows])
        budget = int(np.ceil(len(rows)/10))
        r['new_common_ranking'] = {}
        for name,priority in zip(['calibrated_prior','calibrated_outside','dry_expert'],risk.T,strict=True):
            top = rows[np.lexsort((rows,-priority))[:budget]]
            r['new_common_ranking'][name] = dict(auc=float(roc_auc_score(wrong[rows],priority)), reviewed=budget,
                                                  errors_found=int(wrong[top].sum()), max_risk=float(priority.max()))
        report[tag] = r
        print(tag, {n:(v['fixed'],v['harmed']) for n,v in r['new_common'].items() if v['fixed']+v['harmed']}, flush=True)
    (lib/'domain_outside_witnesses.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
