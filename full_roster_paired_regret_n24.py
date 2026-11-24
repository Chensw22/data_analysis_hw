"""Learn a three-model blend from relative losses on the same passengers under two refits."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference


def decision(p, anchor, w):
    # The incumbent's explicit corrections take priority under either validation design.
    free = (p[:,0]>=.5)==anchor
    return np.where(free,p@w>=.5,anchor)


def regret_inputs(p, baseline, y):
    oriented = np.where(y[:,None],p,1-p).clip(1e-6,1.)
    free = (p[:,0]>=.5)==baseline
    # Unit likelihood makes relative loss and its directional derivative zero on locked rows.
    oriented[~free] = 1.
    return oriented


def weights(left, right, mode):
    if mode=='equal':
        return np.ones(3)/3
    def objective(w):
        if mode=='historical':
            matrices = [left]
        elif mode=='eightfold':
            matrices = [right]
        else:
            matrices = [left,right]
        like = np.stack([m@w for m in matrices],axis=1).clip(1e-8,1)
        regret = np.stack([np.log(m[:,0]) for m in matrices],axis=1)-np.log(like)
        if mode in ('paired_worst','shuffled_worst'):
            scale=.08
            norm=logsumexp(regret/scale,axis=1,keepdims=True)
            multiplier=np.exp(regret/scale-norm)
            loss=scale*(norm[:,0]-np.log(len(matrices)))
        else:
            multiplier=np.full_like(regret,1/len(matrices))
            loss=regret.mean(1)
        grad=-sum(m.T@(multiplier[:,j]/like[:,j]) for j,m in enumerate(matrices))
        return loss.sum()+.5*np.sum((w-np.array([1.,0.,0.]))**2), grad+w-np.array([1.,0.,0.])
    sol=minimize(objective,np.array([1.,0.,0.]),jac=True,method='SLSQP',bounds=[(0,1)]*3,
                 constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(3))],
                 options={'ftol':1.e-8,'maxiter':160})
    assert sol.success,sol.message
    return sol.x


def run():
    _,y,g,rows,oldfold,newfold,champion,gate,cold,_,_,old,new=aligned()
    raw,all_y,groups,fold=data()
    names,half,full=banks(raw,all_y,fold)
    ref=latest_reference(all_y,groups,fold,names,half,full)
    scores=full[:,[names.index(k) for k in ['mitra96','v44_without_name','tabicl']]]
    additional=~np.isin(np.arange(len(all_y)),rows)
    assert additional.sum()==1739
    left,right=regret_inputs(old,champion,y),regret_inputs(new,gate,y)
    def train_weights(t,mode):
        train_right=right[t]
        if mode=='shuffled_worst':
            # Preserve outcomes and the two reference decisions while breaking identity.
            rng=np.random.default_rng(1248)
            partner=np.arange(len(t))
            strata=4*y[t].astype(int)+2*champion[t].astype(int)+gate[t].astype(int)
            for key in np.unique(strata):
                ix=np.flatnonzero(strata==key)
                partner[ix]=rng.permutation(ix)
            train_right=train_right[partner]
        return weights(left[t],train_right,mode)
    report={}
    for method in ['equal','historical','eightfold','pooled','paired_worst','shuffled_worst']:
        pred_old,pred_new=champion.copy(),gate.copy()
        extra=ref.copy()
        hits=np.zeros(len(y),int)
        for f in np.unique(newfold):
            for h in np.unique(oldfold):
                # All base predictions used here exclude all labels of this intersection.
                cell=np.flatnonzero((newfold==f)&(oldfold==h))
                for a,b in GroupKFold(4,shuffle=True,random_state=462).split(cell,groups=g[cell]):
                    t,q=cell[a],cell[b]
                    assert not np.isin(g[q],g[t]).any()
                    w=train_weights(t,method)
                    pred_old[q]=decision(old[q],champion[q],w)
                    pred_new[q]=decision(new[q],gate[q],w)
                    hits[q]+=1
            t=np.flatnonzero(newfold==f)
            # Some groups straddle the historical development roster. Exclude their donors instead.
            q=np.flatnonzero(additional & (fold==f))
            t=t[~np.isin(g[t],groups[q])]
            assert not np.isin(g[t],groups[q]).any()
            extra[q]=decision(scores[q],ref[q],train_weights(t,method))
        np.testing.assert_array_equal(hits,np.ones(len(y),int))
        def summarize(pred,base,truth,grp,split):
            d=(pred==truth).astype(int)-(base==truth)
            return dict(accuracy=float((pred==truth).mean()),fixed=int((d==1).sum()),harmed=int((d==-1).sum()),
                        fold_net=[int(d[split==f].sum()) for f in np.unique(split)],ci_pp=interval(d,grp))
        report[method]=dict(historical=summarize(pred_old,champion,y,g,oldfold),
                native_cold=summarize(pred_old[cold],champion[cold],y[cold],g[cold],oldfold[cold]),
                eightfold_common=summarize(pred_new,gate,y,g,newfold),additional=summarize(extra[additional],
                    ref[additional],all_y[additional],groups[additional],fold[additional]),
                corrected_both=int(((pred_old==y)&(pred_new==y)&(champion!=y)&(gate!=y)).sum()),
                harmed_both=int(((pred_old!=y)&(pred_new!=y)&(champion==y)&(gate==y)).sum()),
                all_development_weights=train_weights(np.arange(len(y)),method).tolist())
        print(method,[(k,report[method][k]['fixed'],report[method][k]['harmed'])
                      for k in ['historical','eightfold_common','additional']],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/paired_regret.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
