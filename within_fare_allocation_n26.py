"""Condition on narrow fare levels when learning which allocation of a fixed bill predicts transport."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import assignments
from cross_deck_budget_multiples_n26 import fields
from full_roster_booking_validation_n23 import ROOT,data
from latent_bill_rounding_n26 import BILLS
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW,sources


def structure(raw):
    total,n,paid,planet,deck=fields(raw)
    bill=raw[BILLS].fillna(0).to_numpy()
    shares=bill/np.maximum(total[:,None],1)
    core=np.column_stack([np.sqrt(shares),shares**2,np.log1p(bill)/6,bill>0])
    pair=np.column_stack([np.sqrt(shares[:,i]*shares[:,j]) for i in range(5) for j in range(i)])
    x=np.column_stack([core,pair,core[:,:5]*(planet=='Mars')[:,None],core[:,:5]*(planet=='Europa')[:,None]])
    key=np.char.add(np.char.add(planet.astype(str),'/'),deck.astype(str))
    return 2*total+n,paid,key,x


def neighbors(t,q,g,cost,paid,key,radius):
    out=[np.empty(0,int) for _ in q]
    for category in np.unique(key[q[paid[q]]]):
        bank=t[paid[t]&(key[t]==category)]
        bank=bank[np.argsort(cost[bank],kind='stable')]
        members=np.flatnonzero(paid[q]&(key[q]==category))
        a=np.searchsorted(cost[bank],cost[q[members]]-radius)
        b=np.searchsorted(cost[bank],cost[q[members]]+radius,side='right')
        for i,lo,hi in zip(members,a,b,strict=True):
            window=bank[lo:hi]
            window=window[g[window]!=g[q[i]]]
            order=np.argsort(abs(cost[window]-cost[q[i]]),kind='stable')
            ranked=window[order]
            _,first=np.unique(g[ranked],return_index=True)
            out[i]=ranked[np.sort(first)[:24]]
    return out


def fit(t,nbr,y,x,base,g):
    matches=set()
    for i,others in zip(t,nbr,strict=True):
        for j in others[y[others]!=y[i]]:
            matches.add((int(i),int(j)) if y[i] else (int(j),int(i)))
    if not matches:return np.zeros(x.shape[1]),0
    left,right=np.array(sorted(matches)).T
    assert np.all(g[left]!=g[right]) and np.all(y[left]>y[right])
    degree=np.bincount(np.r_[left,right],minlength=len(y))
    w=1/np.sqrt(degree[left]*degree[right])
    gap=base[left]-base[right]
    dx=x[left]-x[right]
    def loss(beta):
        value=gap+dx@beta
        return w@np.logaddexp(0,-value)+3*(beta@beta), dx.T@(w*(expit(value)-1))+6*beta
    result=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    assert result.success
    return result.x,len(left)


def run():
    raw,y,g,_=data()
    _,_,_,views=sources()
    old,cur=views['historical'],views['full']
    cost,paid,key,x=structure(raw)
    sets=list(assignments(old,cur,g))
    all_old=np.flatnonzero(old['known'])
    fresh=cur['known']&~np.isin(g,g[old['ids']])
    records={}
    for protocol,parts in [('bookingmix',sets),('whole_history_on_new',[(0,0,all_old,np.flatnonzero(fresh))])]:
        pred={};has={};coverage=np.zeros(len(y),bool);parent=cur['anchor'].copy();fitted=[]
        for f,k,t,q in parts:
            donors=old['ids'][t]
            assert not np.isin(g[donors],g[q]).any()
            if protocol=='bookingmix':
                file=REVIEW/f'pred_bookingmix_{f}_{k}.npz'
            else:file=REVIEW/'pred_historical_to_new.npz'
            with np.load(file) as z:
                np.testing.assert_array_equal(z['q'],q)
                parent[q]^=(z['identities_error']>.5)&(z['identities_outcome']>.5)
            base=np.zeros(len(y))
            base[donors]=old['off'][t]*np.where(old['anchor'][t],-1,1)
            base[q]=cur['off'][q]*np.where(cur['anchor'][q],-1,1)
            for mode,width in [('near',5),('wide',96)]:
                tr=neighbors(donors,donors,g,cost,paid,key,width)
                coef,count=fit(donors,tr,y,x,base,g)
                fitted.append(dict(source_fold=int(f),bucket=int(k),mode=mode,discordant_pairs=count))
                near=neighbors(donors,q,g,cost,paid,key,width)
                steps=np.zeros((len(q),2));support=np.zeros(len(q),int)
                for j,peers in enumerate(near):
                    if not len(peers):continue
                    support[j]=len(peers)
                    w=np.exp(-abs(cost[peers]-cost[q[j]])/(width/2))
                    own=x[q[j]]@coef
                    ref=x[peers]@coef
                    center=np.average(ref,weights=w)
                    steps[j,0]=(own-center)*w.sum()/(3+w.sum())
                    intercept=0.
                    for _ in range(5):
                        pr=expit(base[peers]+ref-center+intercept)
                        intercept+=((w@(y[peers]-pr)-3*intercept)/(w@(pr*(1-pr))+3)).clip(-1,1)
                    steps[j,1]=steps[j,0]+intercept
                has.setdefault(mode,np.zeros(len(y),bool))[q]=support>0
                for j,n in enumerate(('slope','with_level')):
                    adjusted=base[q]+steps[:,j]>=0
                    # This tests the new conditional information in orders whose original probability supports the decision.
                    allow=(base[q]>=0)==parent[q]
                    pred.setdefault(mode+'_'+n,np.zeros(len(y),bool))[q]=allow&(adjusted!=parent[q])&(support>0)
            coverage[q]=True
        wrong=parent!=y
        scopes={'all':coverage,'unanimous':coverage&cur['common'],'fresh':coverage&fresh}
        scores={n:{s:ticker(m,wrong,edit) for s,m in scopes.items()} for n,edit in pred.items()}
        records[protocol]=dict(support={n:int((coverage&m).sum()) for n,m in has.items()},
                               fits=fitted,actions=scores)
        print(protocol,records[protocol]['support'],[(n,d['all']['fixed'],d['all']['harmed'],d['unanimous']['fixed'],d['unanimous']['harmed'])
                                                    for n,d in scores.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/within_fare_allocation.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
