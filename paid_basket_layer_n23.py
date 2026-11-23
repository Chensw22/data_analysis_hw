"""Learn contributions of overlapping spending across service subsets on all payers."""
from itertools import combinations

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development


def matrices(frame, train):
    expenses = frame[BILLS].to_numpy(float)
    amount = np.nan_to_num(expenses)
    total = amount.sum(1)
    registration = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(
        frame.iloc[train][CATEGORICAL]).transform(frame[CATEGORICAL])
    total_log = np.log1p(total)
    # All features are bounded; coefficients on rarely purchased bundles shrink toward zero.
    core = np.c_[registration, frame.Age.fillna(28)/40, frame.Age.isna(),
                 frame.CabinNumber.fillna(850)/1000, frame.CabinNumber.isna(),
                 np.log(frame.BookingSize), np.isnan(expenses), total_log/7,
                 np.maximum(total_log[:,None] - np.log([100,500,1000,3000]),0)/3]
    order1, higher = [], []
    for degree in range(1,6):
        for idx in combinations(range(5),degree):
            shared = amount[:,idx].min(axis=1)
            # The common spending layer is precisely the smallest of the member amounts.
            # Monotone transformations of this minimum add thresholds and budget dependence.
            out = order1 if degree == 1 else higher
            out.append(np.c_[shared>0, np.log1p(shared)/6, shared/np.maximum(total,1)])
    add = np.column_stack([core,*order1])
    return dict(total=core, additive=add, layers=np.column_stack([add,*higher]))


def main():
    row,y,frame,g,_ = load()
    ids,labels,masks = historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    _, release,_ = development()
    paid = strata(frame)==2
    multi = frame[BILLS].gt(0).sum(axis=1).to_numpy()>=2
    masks = {**masks,'paid_repeat':masks['repeat']&paid&y,
             'multi_paid_repeat':masks['repeat']&multi&y,
             'single_paid_repeat':masks['repeat']&paid&~multi&y}
    out = ART/'nov23_paid_basket_layer'
    out.mkdir(exist_ok=True)
    plans,hist,refs = {},{},[]
    negatives=[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as cache:
            for key,value in [('row',row),('y',y)]:
                np.testing.assert_array_equal(cache[key],value)
            fold,base,ref = cache['fold'],cache['base'],cache['ref']
        np.testing.assert_array_equal(ref,release[131] if seed=='v131' else base>=.5)
        signals = {key:np.zeros(len(y)) for key in ['total','additive','layers']}
        for f in np.unique(fold):
            q = np.flatnonzero((fold==f)&paid)
            t = np.flatnonzero((fold!=f)&paid&~np.isin(g,g[fold==f]))
            assert not np.intersect1d(g[t],g[q]).size
            for key,x in matrices(frame,t).items():
                fit = LogisticRegression(C=1.,max_iter=600).fit(x[t],y[t])
                signals[key][q] = fit.decision_function(x[q])
        votes = {}
        eligible = paid & (ref==(base>=.5))
        for key,z in signals.items():
            for name in ['direct','combined','rescue']:
                vote = ref.copy()
                if name=='direct':
                    vote[eligible] = z[eligible]>=0
                else:
                    active = eligible if name=='combined' else eligible&~ref
                    x = np.c_[logit(base.clip(.002,.998)),z]
                    for f in np.unique(fold):
                        ix=np.flatnonzero((fold==f)&active)
                        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                            t,q=ix[a],ix[b]
                            assert not np.intersect1d(g[t],g[q]).size
                            vote[q]=calibrated(x[t],x[q],y,t)>=.5
                votes[name+'_'+key]=vote
        metrics={}
        anchor=ref==y
        if seed!='v131':
            refs.append(anchor)
            negatives.append(~ref)
        for key,v in votes.items():
            h=v==y
            gain=h.astype(int)-anchor
            metrics[key]=dict(accuracy=float(h.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                              paid_false_negatives_fixed=int(((gain>0)&paid&y).sum()),
                              paid_true_negatives_harmed=int(((gain<0)&paid&~y).sum()))
            if seed!='v131':hist.setdefault(key,[]).append(h)
        plans[seed]=metrics
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**signals)
    stable_negative=np.logical_and.reduce(negatives)
    count=frame[BILLS].gt(0).sum(axis=1).to_numpy()
    census={}
    for k in range(1,6):
        cell=stable_negative & (count==k)
        census[str(k)]=dict(cases=int((cell&y).sum()),controls=int((cell&~y).sum()))
    write(ROOT/'reports/nov23_paid_basket_layer/results.json',dict(plans=plans,census=census,
          sizes={k:int(v.sum()) for k,v in masks.items()},grouped={
          k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
