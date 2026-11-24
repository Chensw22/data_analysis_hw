"""Measure attainable and held-out gains from routing the six new predictions."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_solo_coverage_n24 import cached
from full_roster_disagreement_calibration_n24 import stats


def evaluate(probs, base, soft, y, g, folds):
    disagree = (probs >= .5) != base[:,None]
    free = base == (soft >= .5)
    can_change = free & disagree.any(1)
    margin = np.abs(soft-.5)
    # Codes are fixed before looking at outcomes. Refinements retain all previous keys.
    risk_bin = np.digitize(margin, [.1,.2,.3,.4])
    root = base.astype(int)*5+risk_bin
    family = (4*disagree[:,:3].sum(1)+disagree[:,3:].sum(1)).astype(int)
    middle = root*16+family
    pattern = disagree @ (2**np.arange(6))
    leaf = root*64+pattern
    error = base != y
    votes = {'majority': can_change & (disagree.sum(1)>3)}
    bounds = {}
    for name,key in [('risk',root),('family_counts',middle),('six_votes',leaf)]:
        n = np.bincount(key[can_change])
        e = np.bincount(key[can_change],weights=error[can_change],minlength=len(n))
        bounds[name] = dict(nonempty=int(np.count_nonzero(n)),
                            oracle_net=int(np.maximum(2*e-n,0).sum()))
    for f in np.unique(folds):
        ix = np.flatnonzero(folds==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=474).split(ix,groups=g[ix]):
            train,ask = ix[a],ix[b]
            assert not np.isin(g[train],g[ask]).any()
            train,ask = train[can_change[train]],ask[can_change[ask]]
            prior = .5-margin[ask]
            for name,key,width in [('risk', root,10),('family_counts',middle,160),('six_votes',leaf,640)]:
                counts = np.bincount(key[train],minlength=width)[key[ask]]
                misses = np.bincount(key[train],weights=error[train],minlength=width)[key[ask]]
                prior = (misses+16*prior)/(counts+16)
                votes.setdefault(name,np.zeros(len(y),bool))[ask] = prior>.5
    return {k:base ^ v for k,v in votes.items()}, bounds, can_change


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('y',y),('fold',fold),('passenger',raw.PassengerId)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    report = {}
    masks = []
    for tag,ids,f,base,p in [('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                              ('historical',rows,of,champ,old[:,0])]:
        v = np.column_stack([cached(folder,tag,ids,f,g,y) for folder in
                             ('response_channels','broad_error_transport_projection')])
        correct = (v>=.5)==y[ids,None]
        decisions,bounds,eligible = evaluate(v,base,p,y[ids],g[ids],f)
        wrong = base!=y[ids]
        source = correct[:,:3].any(1)
        proj = correct[:,3:].any(1)
        coverage = {k:int((wrong&m).sum()) for k,m in
                    [('channel_only',source&~proj),('projection_only',proj&~source),('both',proj&source),
                     ('neither',~(proj|source)),('eligible_wrong',eligible)]}
        hard_wrong = hard[ids] & wrong
        coverage['among_shared_errors'] = int((hard_wrong&(proj|source)).sum())
        coverage['shared_error_denominator'] = int(hard_wrong.sum())
        masks.append(wrong&(source|proj))
        summary = {}
        for name,pred in decisions.items():
            details = stats(pred,base,y[ids],g[ids],f)
            details['shared_fixed'] = int((hard_wrong&(pred==y[ids])).sum())
            if tag=='full':
                new = ~np.isin(g[ids],og)
                details['new_orders_net'] = int((pred[new]==y[ids][new]).sum()-(base[new]==y[ids][new]).sum())
            summary[name] = details
        report[tag] = dict(coverage=coverage,bounds=bounds,baseline_correct=int((~wrong).sum()),decisions=summary)
        print(tag,coverage,bounds,[(k,s['net']) for k,s in summary.items()],flush=True)
    report['any_correct_both_protocols'] = int((masks[0][rows]&masks[1]).sum())
    (ROOT/'reports/nov23_matched_learning_curve/six_model_rescue.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
