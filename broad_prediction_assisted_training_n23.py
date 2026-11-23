"""Compare truly acquired labels, teacher labels, and bias-corrected use of unlabeled manifests."""
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import load_roster
from broad_boosting_path_n23 import calibrated
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

REPORT = ROOT/'reports/nov23_training_information'
OUT = ART/'nov23_training_information'


def evidence(frame, y, train, query_ids):
    model=CatBoostClassifier(iterations=360,depth=5,learning_rate=.07,thread_count=2,
                             random_seed=219,verbose=False,allow_writing_files=False)
    model.fit(frame.iloc[train],y,cat_features=CATEGORICAL)
    data=Pool(frame.iloc[query_ids],cat_features=CATEGORICAL)
    offset=model.predict(data,prediction_type='RawFormulaVal')
    leaves=model.calc_leaf_indexes(data)[:,::12]
    # Fixed categories avoid using held-out passengers to choose feature vocabularies.
    enc=OneHotEncoder(categories=[np.arange(32) for _ in range(leaves.shape[1])],sparse_output=True)
    enc.fit(np.zeros((1,leaves.shape[1]),int))
    x=hstack([csr_matrix(np.ones((len(leaves),1))),csr_matrix(offset[:,None]),enc.transform(leaves)],format='csr')
    return x,offset


def residual_fit(x,offset,labelled,y,extra,method,extra_labels=None):
    a,b=x[labelled],x[extra]
    off_a,off_b=offset[labelled],offset[extra]
    n,m=len(labelled),len(extra)
    obs=expit(off_a)
    proxy=expit(off_b)
    # (1-lambda)*E_lab[log partition] + lambda*E_extra[log partition]
    # minus a linear response moment. Only 'true' may inspect the extra responses.
    if method=='single':
        weight=0.
    elif method=='true':
        weight=m/(m+n)
        proxy=extra_labels
    else:
        weight=.5
    linear_a=y-weight*obs if method=='corrected' else (1-weight)*y
    target=a.T@linear_a/n+weight*(b.T@proxy)/m
    ridge=.004
    def loss(beta):
        za,zb=off_a+a@beta,off_b+b@beta
        # Constant offset terms don't affect the optimizer.
        value=(1-weight)*np.logaddexp(0,za).mean()+weight*np.logaddexp(0,zb).mean()-target@beta
        grad=(1-weight)*(a.T@expit(za))/n+weight*(b.T@expit(zb))/m-target
        return value+ridge*(beta@beta)/2,grad+ridge*beta
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=250,ftol=1.e-12,gtol=1.e-8))
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x


def fold_scores(frame, row, y, group, t, q, external, f):
    a,bc=next(GroupKFold(2,shuffle=True,random_state=682+f).split(t,groups=group[row[t]]))
    a,bc=t[a],t[bc]
    b,c=next(GroupKFold(2,shuffle=True,random_state=888+f).split(bc,groups=group[row[bc]]))
    b,c=bc[b],bc[c]
    groups=[group[row[s]] for s in (a,b,c,q)]+[group[external]]
    for i in range(len(groups)):
        for j in range(i):assert not np.intersect1d(groups[i],groups[j]).size
    ids=np.r_[row[b],row[c],external,row[q]]
    x,off=evidence(frame,y[a],row[a],ids)
    ib,ic,iu,iq=np.split(np.arange(len(ids)),np.cumsum([len(b),len(c),len(external)]))
    preds={'teacher':expit(off[iq])}
    for name,lab,labels,add,known in [
        ('development',ib,y[b],ic,y[c]),
        ('competition',np.r_[ib,ic],y[np.r_[b,c]],iu,None)]:
        methods=['single','naive','corrected']+(['true'] if known is not None else [])
        for method in methods:
            beta=residual_fit(x,off,lab,labels,add,method,extra_labels=known if method=='true' else None)
            preds[name+'_'+method]=expit(off[iq]+x[iq]@beta)
    # 'development_true' and 'competition_single' use exactly the same true labels and ridge.
    np.testing.assert_allclose(preds['development_true'],preds['competition_single'],atol=.00015,rtol=0)
    return preds,[len(a),len(b),len(c),len(external),len(q)]


def main():
    row,y,masks=historical()
    labels,releases,_=development()
    np.testing.assert_array_equal(labels,y)
    frame,group,size,coverage=load_roster()
    frame['BookingSize']=size
    external=np.arange(len(frame)-coverage['test_rows'],len(frame))
    state=strata(frame.iloc[row])
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    OUT.mkdir(exist_ok=True)
    plans,hist,refs={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        sizes,raw=[],{}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(group[row],group[row[q]]))
            probs,counts=fold_scores(frame,row,y,group,t,q,external,int(f))
            sizes.append(counts)
            for k,v in probs.items():raw.setdefault(k,np.zeros(len(y)))[q]=v
        decisions={k:v>=.5 for k,v in raw.items()}
        for k in ['competition_single','competition_naive','competition_corrected']:
            x=np.c_[logit(base.clip(.002,.998)),logit(raw[k].clip(.002,.998))]
            vote=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=group[row[ix]]):
                    t,q=ix[a],ix[b]
                    vote[q]=calibrated(x[t],x[q],y,t)>=.5
            decisions['combined_'+k]=vote
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        metrics,paired={},{}
        for k,v in decisions.items():
            h=v==y
            metrics[k]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()))
            if seed!='v131':hist.setdefault(k,[]).append(h)
            if k not in ['teacher','development_single','competition_single','combined_competition_single']:
                prefix=k.rsplit('_',1)[0]
                old=decisions[prefix+'_single']==y
                paired[k]=dict(fixed=int((h&~old).sum()),harmed=int((~h&old).sum()),
                               repeat_fixed=int((h&~old&masks['repeat']).sum()),repeat_harmed=int((~h&old&masks['repeat']).sum()))
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**raw)
        plans[seed]=dict(models=metrics,paired=paired,sizes=sizes)
        print(seed,paired['development_true'],paired['competition_corrected'],flush=True)
    write(REPORT/'assisted.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
