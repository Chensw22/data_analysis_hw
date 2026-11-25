"""Test expert-by-expert ordering of a passenger against companions in held bookings."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from held_expert_portfolios_n26 import ROOT, sources, outcome_scores, panel_predictions
from outside_panel_coverage_n26 import projection
from solo_archive_recovery_n26 import ticker


def design(scores):
    logits=logit(scores.clip(.003,.997))
    # Keep the 18 recorded family representatives and three distinct new input views.
    bank=logits[:,[*range(18),89,93,96]]
    own=np.column_stack([np.quantile(logits,[.1,.25,.5,.75,.9],axis=1).T,
                         (logits>=0).mean(1),bank])
    return own,bank


def contrast(own,bank,ids,groups):
    # Pair like-ranked members for the placebo. Every recipient remains in its own
    # fold; donor parties have equal size, disjoint identities, and a similar party mean.
    parties={}
    for _,loc in pd.Series(ids,index=groups[ids]).groupby(level=0):
        members=loc.to_numpy()
        members=members[np.lexsort((members,own[members,2]))]
        parties.setdefault(len(members),[]).append(members)
    width=2*bank.shape[1]+6
    actual=np.zeros((len(ids),width)); sham=np.zeros_like(actual)
    lookup={int(v):i for i,v in enumerate(ids)}
    rng=np.random.default_rng(21046)
    for size,collection in parties.items():
        if size<2:
            continue
        roster=np.array(collection)
        ordered=np.argsort(own[roster,2].mean(1),kind='stable')
        for block in np.array_split(ordered,max(1,len(ordered)//8)):
            cycle=rng.permutation(block)
            donors=np.roll(cycle,1)
            assert len(block)==1 or not np.any(cycle==donors)
            for k,j in zip(cycle,donors,strict=True):
                this,other=roster[k],roster[j]
                alternatives=[(actual,this)]
                if k!=j:
                    assert groups[this[0]]!=groups[other[0]]
                    alternatives.append((sham,other))
                for rank,row in enumerate(this):
                    for out,members in alternatives:
                        peers=np.delete(members,rank)
                        gap=bank[row]-bank[peers]
                        mean=gap.mean(0)
                        relative=mean-np.median(mean)
                        votes=(gap>0).mean(0)
                        peer_scores=own[peers,2]
                        out[lookup[row]]=np.r_[relative,votes-votes.mean(),mean.mean(),mean.std(),
                                           votes.mean(),peer_scores.mean(),peer_scores.std(),len(peers)]
    return actual,sham


def calibrate(xt,xq,y):
    scale=StandardScaler().fit(xt)
    tr,te=[np.clip(scale.transform(a),-5,5) for a in (xt,xq)]
    return LogisticRegression(C=.04,max_iter=500).fit(tr,y).decision_function(te)


def run():
    _,y,g,fold,views,votes=projection(include_full=True)
    v=sources()[-1]['full']
    scores=outcome_scores(v,panel_predictions(v,'full'))
    np.testing.assert_array_equal(scores>=.5,votes)
    own,bank=design(scores)
    party=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    report={}
    for tag in ('full','bookingmix','historical_to_new'):
        valid,parent=views[tag]
        logits=np.full((len(y),3),np.nan)
        # Fit the secondary model exclusively on other orders in the same base fold.
        for k in np.unique(fold):
            rows=np.flatnonzero((fold==k)&(party>1))
            if tag=='historical_to_new':
                fresh=np.isin(g[rows],g[valid])
                parts=[(rows[~fresh],rows[fresh])]
            else:
                parts=[(rows[t],rows[q]) for t,q in GroupKFold(4,shuffle=True,random_state=5104).split(rows,groups=g[rows])]
            for t,q in parts:
                if not len(q):continue
                assert not np.isin(g[t],g[q]).any()
                tr,tc=contrast(own,bank,t,g)
                va,vc=contrast(own,bank,q,g)
                for j,(a,b) in enumerate([(None,None),(tr,va),(tc,vc)]):
                    xt=own[t] if a is None else np.column_stack([own[t],a])
                    xq=own[q] if b is None else np.column_stack([own[q],b])
                    logits[q,j]=calibrate(xt,xq,y[t])
        mask=valid&(party>1)
        assert np.isfinite(logits[mask]).all()
        flip=(logits>=0)!=parent[:,None]
        actions={name:flip[:,j] for j,name in enumerate(['individual','companions','placebo'])}
        actions['only_companions']=flip[:,1]&~flip[:,0]
        actions['only_companions_vs_both']=flip[:,1]&~flip[:,0]&~flip[:,2]
        for j,name in enumerate(['individual','companions','placebo']):
            swap=np.zeros(len(y),bool)
            for _,pos in pd.Series(np.flatnonzero(mask),index=g[mask]).groupby(level=0):
                rows=pos.to_numpy()
                if len(rows)!=party[rows[0]]:
                    continue
                count=parent[rows].sum()
                if 0<count<len(rows):
                    ordered=rows[np.lexsort((rows,-logits[rows,j]))]
                    swap[ordered[:count]]=~parent[ordered[:count]]
                    swap[ordered[count:]]=parent[ordered[count:]]
            actions['preserve_count_'+name]=swap
        actions['preserve_count_real_only']=actions['preserve_count_companions']&~actions['preserve_count_individual']&~actions['preserve_count_placebo']
        wrong=parent!=y
        strict=(votes==parent[:,None]).all(1)
        r={s:{n:ticker(m,wrong,flag) for n,flag in actions.items()}
           for s,m in [('multiple',mask),('unanimous_multiple',mask&strict),('contested_multiple',mask&~strict)]}
        r['opportunities']=int(mask.sum())
        report[tag]=r
        print(tag,[(n,o['fixed'],o['harmed']) for n,o in r['multiple'].items()],flush=True)
    assert report['full']['multiple']['individual']['errors']==580
    (ROOT/'reports/nov23_matched_learning_curve/within_booking_vote_order.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
