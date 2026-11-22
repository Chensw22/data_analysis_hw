"""Bidirectional correction across the whole roster, with explicit uncertainty over rank cutoffs."""
import numpy as np
from scipy.special import expit,logit,logsumexp
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from broad_bill_exception_n23 import strata
from broad_paid_recovery_budget_n23 import ART,REPORT,SEEDS,historical,load,prepare,write,ranks,best,options,applied


def evidence(scores_t,scores_q,risk_t,risk_q,errors):
    # Include the unmodified baseline with half the prior mass. For each
    # possible correction, pooled observations in its score-ranked prefix
    # can increase the odds of an error. Integrate over ranker, cutoff and odds.
    shifts=np.array([.5,1.,2.,4.])
    adjust=expit(logit(risk_q)[:,None]+shifts)-risk_q[:,None]
    z=logit(risk_t)
    logratios=errors[:,None]*shifts-np.logaddexp(0,z[:,None]+shifts)+np.logaddexp(0,z)[:,None]
    weights=[np.array([np.log(.5)])]
    changes=[np.zeros((len(risk_q),1))]
    quant=np.linspace(0,1,21)
    logprior=np.log(.5/(scores_t.shape[1]*2*len(quant)*len(shifts)))
    for a,b in zip(scores_t.T,scores_q.T):
        for direction in (1,-1):
            x=direction*a
            cuts=np.quantile(x,quant)
            mask=x[:,None]>=cuts
            ll=mask.T@logratios
            weights.append(ll.ravel()+logprior)
            changes.append(((direction*b[:,None]>=cuts)[:,:,None]*adjust[:,None,:]).reshape(len(b),-1))
    w=np.concatenate(weights)
    w=np.exp(w-logsumexp(w))
    return risk_q+np.column_stack(changes)@w


def enriched(name,row,y):
    keys,values,fold,base,ref=ranks(name,row,y)
    for src,k in (("broad_neural_path","epoch_80"),("broad_boosting_path","trees_800"),
                  ("broad_training_membership","ordinary")):
        with np.load(ART/f"nov23_{src}/{name}.npz") as z:
            for label,v in (("row",row),("y",y),("fold",fold),("base",base),("ref",ref)):
                np.testing.assert_array_equal(z[label],v)
            values=np.column_stack([values,z[k]])
            keys.append(src+"/"+k)
    return keys,values,fold,base,ref


def main():
    row,y,frame,group,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    dom=prepare(frame)[0]
    paid=dom>=0
    state=strata(frame)
    basket=np.where(paid,dom,np.where(state==0,5,6))
    repeat_paid=masks["repeat"]&paid&y
    repeat_zero=masks["repeat"]&(state==0)&~y
    scopes=dict(**masks,repeat_paid=repeat_paid,repeat_zero=repeat_zero)
    summary,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        keys,vs,fold,base,ref=enriched(name,row,y)
        full=np.array([i for i,k in enumerate(keys) if not k.startswith(("paid_register","broad_service_transfer"))])
        eligible=ref==(base>=.5)
        err=y!=ref
        risk=np.where(ref,1-base,base).clip(.001,.999)
        cells=basket*2+ref.astype(int)
        actions={k:np.zeros(len(y),bool) for k in ("hindsight","empirical","average","base_average")}
        for cell in np.unique(cells):
            ids=np.flatnonzero((cells==cell)&eligible)
            if not len(ids):
                continue
            values=vs if paid[ids[0]] else vs[:,full]
            _,_,rule=best(options(ids,values,err))
            actions["hindsight"][ids]=applied(ids,rule,values)
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=911).split(ids,groups=group[ids]):
                train,query=ids[a],ids[b]
                assert not np.intersect1d(group[train],group[query]).size
                for cell in np.unique(cells[query]):
                    t=train[(cells[train]==cell)&eligible[train]]
                    q=query[(cells[query]==cell)&eligible[query]]
                    if len(t)<2 or not len(q):
                        continue
                    values=vs if paid[q[0]] else vs[:,full]
                    _,_,rule=best(options(t,values,err))
                    actions["empirical"][q]=applied(q,rule,values)
                    for k,x in (("average",values),("base_average",values[:,[0]])):
                        actions[k][q]=evidence(x[t],x[q],risk[t],risk[q],err[t])>.5
        stats={}
        for k,act in actions.items():
            assert not (act&~eligible).any()
            hit=(ref^act)==y
            stats[k]=dict(fixed=int((err&act).sum()),harmed=int((~err&act).sum()),accuracy=float(hit.mean()),
                          net_by_fold=[int(((hit.astype(int)-(ref==y))[fold==f]).sum()) for f in np.unique(fold)])
            if name!="v131":
                history.setdefault(k,[]).append(hit)
        summary[name]=stats
        print(name,stats,flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in history.items()}
    write(REPORT/"whole_roster.json",dict(plans=summary,stable=stable,rankers=keys))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
