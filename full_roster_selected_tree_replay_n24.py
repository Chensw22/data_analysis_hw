"""Retest rare dissent from a feature-selected tree ensemble after changing training orders."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_kernel_repair_overlap_n24 import from_cache
from full_roster_permutation_selection_n24 import OUT, ensemble, feature_bank, select


def refit(raw,y,g,seed):
    x,fields,core = feature_bank(raw)
    out = ROOT/'artifacts/nov24_selected_tree_replay'
    out.mkdir(exist_ok=True)
    scores = np.full(len(y),np.nan)
    for f,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,groups=g)):
        assert not np.isin(g[t],g[q]).any()
        path = out/f'{seed}_{f}.npz'
        if not path.exists():
            columns = select(x,y,t,g,core,497)[1]
            p = ensemble(x[:,columns],y,t,q,497)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=t,q=q,truth=y[t],fields=fields,columns=columns,p=p)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,v in [('t',t),('q',q),('truth',y[t]),('fields',fields)]:
                np.testing.assert_array_equal(z[k],v)
            scores[q] = z['p']
        print(seed,f,flush=True)
    assert np.isfinite(scores).all()
    return scores


def original_prediction(rows,fold,g,y,tag):
    out = np.empty(len(rows))
    for f in np.unique(fold):
        mask = fold==f
        q = rows[mask]
        t = rows[~mask & ~np.isin(g[rows],g[q])]
        with np.load(OUT/f'{tag}_{f}.npz') as z:
            for k,v in [('query',q),('train',t),('labels',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            out[mask] = z['prob'][:,1]
    return out


def prepared_vote(p,baseline,opinions):
    agreement = ((opinions>=.5)==baseline[:,None]).all(1)
    return np.where(agreement,p>=.5,baseline)


def run():
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/audit.npz') as z:
        for k,v in [('passenger',raw.PassengerId.to_numpy(str)),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        full_opinions = z['scores']
    _,previous = from_cache('historical',rows,ofold,y,g)
    old_opinions = np.column_stack([old,previous])
    idx = [names.index(k) for k in ('mitra96','v44_without_name','tabicl')]
    comparable = np.column_stack([full[:,idx],full_opinions[:,19:23]])
    assert full_opinions.shape[1]==26 and comparable.shape[1]==old_opinions.shape[1]==7
    q = np.arange(len(y))
    orig = original_prediction(q,fold,g,y,'full')
    native = original_prediction(rows,ofold,g,y,'historical')
    first = prepared_vote(orig,ref,full_opinions)
    five = first!=ref
    assert five.sum()==5 and np.all(first[five]==y[five])
    predictions = [refit(raw,y,g,seed) for seed in (1021,1022)]
    scores = {'original':orig,'new_split_1':predictions[0],'new_split_2':predictions[1],
              'new_split_mean':np.mean(predictions,axis=0)}
    report = {'models':{},'historical_matched_seven':stats(prepared_vote(native,champ,old_opinions),champ,oy,og,ofold)}
    for k,p in scores.items():
        votes = prepared_vote(p,ref,full_opinions)
        matched = prepared_vote(p,ref,comparable)
        report['models'][k] = dict(direct=stats(p>=.5,ref,y,g,fold),
            agree_26=stats(votes,ref,y,g,fold),agree_7=stats(matched,ref,y,g,fold),
            recovered_original_five=int((five & (votes==y)).sum()),
            flips=raw.loc[votes!=ref,'PassengerId'].tolist())
        nine_names = [*FAMILIES, 'mitra96', *EVIDENCE]
        nine = full[:,[names.index(name) for name in nine_names]]
        unseen = ~np.isin(g,og)
        assert unseen.sum() == 1044
        for size, opinions in ((3, full[:,idx]), (5, nine[:,:5]), (9, nine)):
            decision = prepared_vote(p,ref,opinions)
            report['models'][k][f'agree_{size}'] = dict(
                full=stats(decision,ref,y,g,fold),
                unseen=stats(decision[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]))
        print(k,[(report['models'][k][s]['corrected'],report['models'][k][s]['harmed'])
                 for s in ('direct','agree_26','agree_7')],report['models'][k]['recovered_original_five'],flush=True)
    both = ((predictions[0]>=.5)!=ref) & ((predictions[1]>=.5)!=ref)
    joint = prepared_vote(np.where(both,1-ref.astype(int),ref),ref,full_opinions)
    report['new_fits_agree'] = stats(joint,ref,y,g,fold)
    report['original_five_repeated_in_both'] = int((five & both).sum())
    path = ROOT/'reports/nov23_matched_learning_curve/selected_tree_replay.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(3):
        run()
