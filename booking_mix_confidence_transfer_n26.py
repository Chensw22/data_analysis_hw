"""Transfer historical confidence calibration to whole unseen orders with the contest's party-size mix."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT


def partition(group, salt, n):
    return np.array([int.from_bytes(hashlib.blake2b((salt+gid).encode(),digest_size=4).digest(),'big')%n
                     for gid in group])


def assignments(old, cur, group):
    oldg=group[old['ids']]
    by_group=pd.DataFrame(dict(g=oldg,fold=old['fold'])).groupby('g').fold
    single=by_group.min().where(by_group.nunique()==1,-1)
    # With no old record there is a free choice of source fold. Choose without consulting outcomes.
    choice=pd.Series(group).map(single).fillna(-2).to_numpy(int)
    new=choice==-2
    choice[new]=partition(group[new],'source:',3)
    shard=partition(group,'query:',5)
    for f in np.unique(old['fold']):
        source=old['fold']==f
        for k in range(5):
            t=np.flatnonzero(source&old['known']&(shard[old['ids']]!=k))
            q=np.flatnonzero((choice[cur['ids']]==f)&(shard[cur['ids']]==k)&cur['known'])
            assert not np.isin(group[cur['ids'][q]],oldg[t]).any()
            # All historical base models producing these source rows trained outside fold f.
            assert not np.isin(group[cur['ids'][q]],oldg[~source]).any()
            yield f,k,t,q


def audit(delta, ids, size, group, target, draws):
    band=np.minimum(size[ids],4)-1
    table={}
    gain=[]
    for s in range(4):
        sub=band==s
        table[str(s+1)]=dict(n=int(sub.sum()),fixed=int((delta[sub]>0).sum()),harmed=int((delta[sub]<0).sum()))
        _,gi,n=np.unique(group[ids][sub],return_inverse=True,return_counts=True)
        total=np.bincount(gi,weights=delta[sub])
        counts=draws[s]
        gain.append((counts@total)/(counts@n))
    center=float(target@np.array([delta[band==s].mean() for s in range(4)]))
    return dict(by_size=table,fixed=int((delta>0).sum()),harmed=int((delta<0).sum()),
                poststratified_gain_pp=100*center,group_bootstrap_pp=(100*np.quantile(target@np.array(gain),[.025,.975])).tolist())


def run():
    from tabicl_consensus_review_n26 import sources
    from stable_confidence_axes_n26 import calibrated, standardized
    from vote_signature_transfer_n26 import transfer as vote_transfer
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    size=raw.booking_size.to_numpy()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    test_group=test.PassengerId.str.split('_').str[0]
    target=np.bincount(test_group.map(test_group.value_counts()).clip(upper=4),minlength=5)[1:]/len(test)
    assert not np.isin(g,test_group).any()
    assert (cur['ids']==np.arange(len(y))).all()
    oldvotes,curvotes=old['votes'],cur['votes']
    family_size=np.ones(curvotes.shape[1])
    families=np.array([s.split('/')[0] for s in cur['names'][18:]])
    for j,name in enumerate(families):
        family_size[18+j]=1/(families==name).sum()
    def native(v):
        return expit(v['off']*np.where(v['anchor'],-1,1))
    decisions={};tested=np.zeros(len(y),bool)
    records=[]
    for f,k,t,q in assignments(old,cur,g):
        assert not tested[q].any()
        tested[q]=True
        joined={n:np.concatenate([old[n][t],cur[n][q]]) for n in ('off','basic','relative','wrong')}
        train=np.arange(len(t)); query=len(t)+np.arange(len(q))
        x=standardized(joined,train)
        anonymous=calibrated(joined,joined,train,query)
        named=calibrated(joined,joined,train,query,x,x)
        plain=native(old)
        near=vote_transfer((oldvotes[t],old['anchor'][t],plain[t],y[old['ids'][t]],g[old['ids'][t]]),
                           (curvotes[q],cur['anchor'][q],native(cur)[q]),family_size)
        actions={'distribution':anonymous>0,'named':named>0,'named_only':(named>0)&(anonymous<=0),
                 'hard_votes':near[:,1],'both':(named>0)&near[:,1]}
        for n,flag in actions.items():
            decisions.setdefault(n,np.zeros(len(y),bool))[q]=flag
        records.append(dict(source_fold=int(f),query_bucket=k,source_rows=len(t),target_rows=len(q)))
    iq=np.flatnonzero(tested)
    bands=size[iq].clip(max=4)
    rng=np.random.default_rng(11163)
    draws=[]
    for b in range(1,5):
        n=len(np.unique(g[iq[bands==b]]))
        draws.append(rng.multinomial(n,np.full(n,1/n),size=1500))
    fresh=~np.isin(g,g[old['ids']])
    wrong=cur['wrong']
    report=dict(target_test_share=target.tolist(),known_rows=int(cur['known'].sum()),included=len(iq),
                fresh_rows=int((tested&fresh).sum()),baseline_correct=int((~wrong[iq]).sum()),coverage=[],
                allocations=records,methods={})
    for b in range(1,5):
        in_band=size.clip(max=4)==b
        report['coverage'].append(dict(size_band=b,known=int((cur['known']&in_band).sum()),
            eligible=int((tested&in_band).sum()),completely_new=int((fresh&tested&in_band).sum())))
    for name,change in decisions.items():
        delta=change*np.where(wrong,1,-1)
        result=audit(delta[iq],iq,size,g,target,draws)
        result['unanimous']={k:int(((delta==s)&tested&cur['common']).sum()) for k,s in [('fixed',1),('harmed',-1)]}
        report['methods'][name]=result
    print(report['coverage'],[(n,r['fixed'],r['harmed'],round(r['poststratified_gain_pp'],3)) for n,r in report['methods'].items()])
    (ROOT/'reports/nov23_matched_learning_curve/booking_mix_confidence_transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
