"""Align archived alternatives on six-way consensus and cross-fit a choice of dissenter."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_solo_coverage_n24 import EXPERIMENTS, cached


def prediction_sets(raw, y, g, fold, rows, old_fold):
    with np.load(ROOT/'artifacts/nov24_remaining_cases/archive_scores.npz') as z:
        for key,val in [('row',rows),('old_fold',old_fold),('fold',fold),('passenger',raw.PassengerId),('y',y)]:
            np.testing.assert_array_equal(z[key],val)
        names = z['names'].tolist()
        matrices = {tag:[z[tag]] for tag in ('full','historical')}
    names += [folder+'/'+key for folder,keys in EXPERIMENTS.items() for key in keys]
    sets = []
    for tag,ix,f in [('full',np.arange(len(y)),fold),('historical',rows,old_fold)]:
        a = matrices[tag] + [cached(folder,tag,ix,f,g,y) for folder in EXPERIMENTS]
        alternatives = np.column_stack(a)
        assert alternatives.shape == (len(ix),len(names)) and len(names)==len(set(names))
        six = np.column_stack([cached(k,tag,ix,f,g,y) for k in ('response_channels','broad_error_transport_projection')])
        sets.append((six,alternatives))
    return names,sets


def impact(mask, correct):
    return dict(fixed=int((mask & ~correct).sum()),harmed=int((mask & correct).sum()))


def select_dissenter(eligible, direction, p, prob, truth, group, folds):
    # Require an actual dissent and let the training orders abstain or select one
    # archive column at a specified opposition strength. No query labels enter.
    conf = np.where(direction,p,1-p)
    bucket = 2*direction.astype(int)+(conf>=.8)
    strength = np.where(direction[:,None],1-prob,prob)
    rules = np.column_stack([strength>t for t in (.5,.65,.8)]) & eligible[:,None]
    assert np.isfinite(prob).all()
    delta = np.where(direction==truth,-1,1)
    out = {k:np.zeros(len(p),bool) for k in ('pooled','by_risk','confirmed')}
    for f in np.unique(folds):
        ix = np.flatnonzero(folds==f)
        parts = [ix[b] for _,b in GroupKFold(4,shuffle=True,random_state=184).split(ix,groups=group[ix])]
        for i,q in enumerate(parts):
            fit = np.concatenate([part for j,part in enumerate(parts) if i!=j])
            def winner(t):
                gain = delta[t] @ rules[t].astype(int)
                if gain.max(initial=0)<=0:
                    return None
                return int(gain.argmax())
            choice = winner(fit)
            if choice is not None:
                out['pooled'][q] = rules[q,choice]
            for k in range(4):
                held = q[bucket[q]==k]
                chosen = winner(fit[bucket[fit]==k])
                if chosen is not None:
                    out['by_risk'][held] = rules[held,chosen]
                check = parts[(i+1)%4]
                train = np.concatenate([parts[(i+2)%4],parts[(i+3)%4]])
                chosen = winner(train[bucket[train]==k])
                if chosen is not None:
                    validate = check[bucket[check]==k]
                    if delta[validate] @ rules[validate,chosen].astype(int)>0:
                        out['confirmed'][held] = rules[held,chosen]
            assert not np.isin(group[q],group[fit]).any()
    return out


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,sets = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,val in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],val)
        hard = z['remaining']
    masks = []
    results = {}
    for (tag,ix,f,base,p),(six,alternatives) in zip([
        ('full',np.arange(len(y)),fold,ref,full[:,4]),('historical',rows,of,champ,old[:,0])],sets,strict=True):
        agree = ((six>=.5)==base[:,None]).all(1)
        wrong = agree & (base!=y[ix])
        alt = (alternatives>=.5)!=base[:,None]
        fixes = wrong[:,None] & alt
        harms = (agree & (base==y[ix]))[:,None] & alt
        masks.append((agree,fixes,harms))
        assert wrong.sum()==(848 if tag=='full' else 733)
        counts = fixes.sum(1)
        policies = select_dissenter(agree,base,p,alternatives,y[ix],g[ix],f)
        metrics = {}
        for name,mask in policies.items():
            out = impact(mask,base==y[ix])
            out['on_shared_errors'] = int((mask & wrong & hard[ix]).sum())
            if tag=='full':
                out['new_orders'] = impact(mask & ~np.isin(g[ix],og),base==y[ix])
            metrics[name] = out
        results[tag] = dict(eligible=int(agree.sum()),errors=int(wrong.sum()),any_correct=int((counts>0).sum()),
            no_correct=int((wrong & (counts==0)).sum()),only_one=int((counts==1).sum()),
            models={k:dict(fixed=int(fixes[:,j].sum()),harmed=int(harms[:,j].sum()),
                           exclusive=int((fixes[:,j] & (counts==1)).sum())) for j,k in enumerate(names)},
            policies=metrics)
    (af,ff,hf),(ah,fh,hh) = masks
    persistent = af[rows] & ah & (ref[rows]!=oy) & (champ!=oy)
    repeat = ff[rows] & fh
    unique = repeat.sum(1)==1
    overlap = {k:dict(repeat_fixed=int(repeat[:,j].sum()),repeat_harmed=int((hf[rows,j]&hh[:,j]).sum()),
                      exclusive_repeat=int((repeat[:,j]&unique).sum())) for j,k in enumerate(names)}
    assert (repeat.any(1)<=persistent).all()
    results['both_protocols'] = dict(common_errors=int(persistent.sum()),same_column_correct=int(repeat.any(1).sum()),
        any_column_each=int((ff[rows].any(1)&fh.any(1)).sum()),same_column_only_one=int(unique.sum()),models=overlap)
    path = ROOT/'reports/nov23_matched_learning_curve/six_failure_archive.json'
    path.write_text(json.dumps(results,indent=2)+'\n')
    for tag in ('full','historical'):
        print(tag, {k:v for k,v in results[tag].items() if k!='models'},flush=True)
    print({k:v for k,v in results['both_protocols'].items() if k!='models'})
    print(sorted(overlap.items(),key=lambda r:-r[1]['repeat_fixed'])[:3])


if __name__=='__main__':
    run()
