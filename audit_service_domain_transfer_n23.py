"""Compare service candidates at equal reversal counts across independently scored official domains."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_service_replacement_budget_n23 import allocate,switch
from broad_target_shift_n23 import make_target_mix
from exaone_validation_data import CATEGORICAL,compact_features

REPORT=ROOT/'reports/nov23_service_domain_transfer'


def domain_by_booking():
    # Load features only. No names, outcomes or numerical booking identifier enter the model.
    parts=[pd.read_csv(ROOT/f'data/raw/{s}.csv',usecols=lambda c:c not in ('Transported','Name'))
           for s in ['train','test']]
    roster=pd.concat(parts,ignore_index=True)
    booking=roster.PassengerId.str.split('_').str[0].astype(int)
    domain=np.arange(len(roster))>=len(parts[0])
    assert not np.intersect1d(booking[domain],booking[~domain]).size
    f=compact_features(roster)
    num=f.drop(columns=CATEGORICAL)
    for c in ('FoodCourt','Spa','ShoppingMall','VRDeck','RoomService'):
        num[c]=np.log1p(num[c])
    obs=pd.get_dummies(f[CATEGORICAL],dtype=float)
    x=pd.concat([num.groupby(booking).mean(),num.groupby(booking).std().add_suffix('_sd'),
                 num.isna().groupby(booking).mean().add_suffix('_missing'),obs.groupby(booking).mean()],axis=1)
    size=booking.value_counts()
    x['size']=size.reindex(x.index)
    x['distinct_cabins']=roster.Cabin.groupby(booking).nunique().reindex(x.index)
    d=pd.Series(domain).groupby(booking).first().reindex(x.index).to_numpy()
    score=np.full(len(d),np.nan)
    for t,q in StratifiedKFold(5,shuffle=True,random_state=7731).split(x,d):
        assert not np.intersect1d(x.index[t],x.index[q]).size
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=9,l2_regularization=8,
                                             learning_rate=.055,min_samples_leaf=48,random_state=7731,early_stopping=False)
        model.fit(x.iloc[t],d[t])
        score[q]=model.predict_proba(x.iloc[q])[:,1]
    # Each held-out booking has exactly one domain score. AUC influence values give
    # a booking-level interval conditional on these OOF scores, with midrank ties.
    target,source=score[d],score[~d]
    def compare(values,against):
        a=np.sort(against)
        return (np.searchsorted(a,values,'left')+np.searchsorted(a,values,'right'))/(2*len(a))
    a,b=compare(target,source),1-compare(source,target)
    auc=roc_auc_score(d,score)
    se=np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    np.testing.assert_allclose([a.mean(),b.mean()],auc)
    by_person=pd.Series(score,index=x.index).loc[booking].to_numpy()
    with np.load(ART/'population_shift.npz') as z:
        old=np.asarray(z['weight'])
    assert len(old)==len(roster)
    old=pd.Series(old).groupby(booking).mean().loc[x.index].to_numpy()
    stats=dict(train_bookings=int((~d).sum()),test_bookings=int(d.sum()),group_auc=auc,
               group_auc_conditional95=[auc-1.96*se,auc+1.96*se],
               passenger_auc=roc_auc_score(domain,by_person),
               archive_rank_correlation=spearmanr(score,old).statistic)
    return by_person,domain,booking.to_numpy(),roster.HomePlanet.fillna('missing').to_numpy(dtype=str),stats


def summary(v,old,y):
    a=v==y
    b=old==y
    fixed=int((a&~b).sum())
    harmed=int((~a&b).sum())
    return dict(people=len(y),accuracy=a.mean(),fixed=fixed,harmed=harmed,net=fixed-harmed)


def grouped_contrast(v,g):
    return dict(mean_pp=100*np.mean(v),ci95_pp=interval(v,g))


def weighted_group_contrast(gain,weights,g):
    _,idx=np.unique(g,return_inverse=True)
    num=np.bincount(idx,weights=gain*weights)
    den=np.bincount(idx,weights=weights)
    rng=np.random.default_rng(693)
    result=[]
    for _ in range(50):
        picks=rng.integers(len(num),size=(80,len(num)))
        result.extend(100*num[picks].sum(1)/den[picks].sum(1))
    return dict(mean_pp=100*num.sum()/den.sum(),ci95_pp=np.quantile(result,[.025,.975]).tolist())


def interaction(hi,lo,mask,g):
    # Groups cannot straddle either stratum. Bootstrap each stratum's bookings independently.
    rng=np.random.default_rng(611)
    reps=[]
    for data,ids in [(hi,np.flatnonzero(mask)),(lo,np.flatnonzero(~mask))]:
        _,ix=np.unique(g[ids],return_inverse=True)
        n=np.bincount(ix)
        s=np.bincount(ix,weights=data)
        draws=[]
        for _ in range(50):
            k=rng.integers(len(n),size=(80,len(n)))
            draws.extend(s[k].sum(1)/n[k].sum(1))
        reps.append(np.array(draws))
    return dict(difference_pp=100*(hi.mean()-lo.mean()),ci95_pp=np.quantile(100*(reps[0]-reps[1]),[.025,.975]).tolist())


def main():
    model_score,domain,groups,planet,diag=domain_by_booking()
    row,y,_,g,_=load()
    np.testing.assert_array_equal(groups[row],g)
    assert not domain[row].any()
    cut=np.median(model_score[row])
    high=model_score[row]>=cut
    assert not np.intersect1d(g[high],g[~high]).size
    diag.update(cut=cut,development_high=int(high.sum()),development_low=int((~high).sum()),
                official_test_fraction_high=float((model_score[domain]>=cut).mean()))
    size=pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    mix=make_target_mix(size,planet,row,np.flatnonzero(domain))
    reports,history={},{'high':{},'low':{}}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_service_bag/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            base,ref,fold,linear=z['base'],z['ref'],z['fold'],z['fixed']
        with np.load(ART/f'nov23_service_bag_trees/{seed}.npz') as z:
            for k,expected in [('base',base),('ref',ref),('fold',fold),('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],expected)
            tree=z['initial']
        if seed!='v131':
            for f in np.unique(fold):
                assert not np.intersect1d(g[fold==f],g[fold!=f]).size
        if seed=='v131':
            # Frozen replay, including overrides from this released model.
            assert np.any(ref!=(base>=.5))
        plans={}
        for label,mask in [('high',high),('low',~high)]:
            r,b,f=ref[mask],base[mask],fold[mask]
            n=round(mask.sum()*115/4277)
            forced=r.copy()
            for k,take in allocate(f,n):
                ids=np.flatnonzero((f==k)&(r==(b>=.5)))
                ids=ids[np.argsort(abs(b[ids]-.5),kind='stable')[:take]]
                forced[ids]=~forced[ids]
            candidates={'reference':r,'base_uncertain':forced}
            for name,p in [('tree',tree[mask]),('linear',linear[mask])]:
                candidates[name]=switch(b,r,p,f,n)[0]
                candidates[name+'_quarter']=np.where(r==(b>=.5),.75*b+.25*p>=.5,r)
                assert np.sum(candidates[name]!=r)==n
            assert np.sum(forced!=r)==n
            plans[label]={k:summary(v,r,y[mask]) for k,v in candidates.items()}
            if seed!='v131':
                for k,v in candidates.items():
                    history[label].setdefault(k,[]).append((v==y[mask]).astype(float))
        weighted={}
        for name,p in [('linear',linear),('tree',tree)]:
            pred=np.where(ref==(base>=.5),.75*base+.25*p>=.5,ref)
            gain=(pred==y).astype(float)-(ref==y)
            weighted[name]=dict(unweighted_gain_pp=100*gain.mean(),official_mix_gain_pp=100*np.average(gain,weights=mix))
        reports[seed]=dict(equal_budget=plans,native_target_mix=weighted)
    comparison={}
    for a,b in [('tree','reference'),('linear','reference'),('base_uncertain','reference'),
                ('tree','linear'),('tree','base_uncertain'),('tree_quarter','reference'),('linear_quarter','reference')]:
        diff={k:np.mean(h[a],axis=0)-np.mean(h[b],axis=0) for k,h in history.items()}
        comparison[a+'-minus-'+b]=dict(high=grouped_contrast(diff['high'],g[high]),
                                      low=grouped_contrast(diff['low'],g[~high]),
                                      high_minus_low=interaction(diff['high'],diff['low'],high,g))
    official={}
    for a,b in [('tree_quarter','reference'),('linear_quarter','reference'),('tree_quarter','linear_quarter')]:
        d=np.empty(len(g))
        for label,m in [('high',high),('low',~high)]:
            d[m]=np.mean(history[label][a],0)-np.mean(history[label][b],0)
        official[a+'-minus-'+b]=weighted_group_contrast(d,mix,g)
    write(REPORT/'results.json',dict(domain=diag,plans=reports,paired_grouped=comparison,official_mix=official))
    print(diag,comparison,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        main()
