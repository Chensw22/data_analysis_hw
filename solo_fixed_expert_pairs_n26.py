"""Upper bounds and held-booking gains from one named dissenter or two specific co-dissenters."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from attentive_tabnet_n26 import MODELS
from expanded_joint_error_witnesses_n26 import FAMILIES
from held_expert_portfolios_n26 import NEW, sources, splits
from outside_panel_coverage_n26 import archives
from solo_remaining_countervotes_n26 import context
from target_mix_expert_routing_n26 import ROOT

REPORT=ROOT/'reports/nov23_matched_learning_curve/solo_fixed_expert_pairs.json'


def exposure_partitions(raw,y,g,fold,ids):
    views=sources()[-1]
    b=np.full(len(y),-1)
    for key,_,v,t,q in splits(views,g):
        if key.startswith('full_'):
            ix=v['ids'][q]
            assert not np.isin(g[ix],g[v['ids'][t]]).any() and (b[ix]<0).all()
            b[ix]=int(key.rsplit('_',1)[1])
    ex=np.full(len(y),3)
    path=ROOT/'artifacts/exaone_group_20261117'
    with np.load(path/'EXAONE_v91_CAST_default8_seed20261112.npz') as z:
        ex[z['dev_ids']]=z['fold_id']
    for k in range(3):
        with np.load(path/f'fold{k}_rows.npz') as z:
            assert not np.isin(g[z['train_ids']],g[ex==k]).any()
            np.testing.assert_array_equal(np.sort(z['validation_ids']),np.flatnonzero(ex==k))
    with np.load(ROOT/'artifacts/nov26_exaone_remaining_orders/scores.npz') as z:
        np.testing.assert_array_equal(z['q'],np.flatnonzero(ex==3))
        assert not np.isin(g[z['t']],g[ex==3]).any()
    assert (b[ids]>=0).all() and len(np.unique(g[ids]))==len(ids)
    return (12*fold[ids]+4*b[ids]+ex[ids]).astype(int)


def choose(count,net,rows):
    objective=net*(rows+1)-count
    winner=int(objective.argmax())
    return winner if objective[winner]>0 else -1


def metrics(act,wrong,fold):
    fixed=act&wrong; harmed=act&~wrong
    d=fixed.astype(int)-harmed
    return dict(fixed=int(fixed.sum()),harmed=int(harmed.sum()),net=int(d.sum()),
                net_by_fold=[int(d[fold==k].sum()) for k in np.unique(fold)])


def scenario(A,names,labels,part,wrong,fold):
    n=len(wrong)
    oracle=np.zeros(n,bool)
    held=oracle.copy()
    cells={}
    signed=np.where(wrong,1,-1)
    for cell in np.unique(labels):
        ids=np.flatnonzero(labels==cell)
        a=A[ids]
        cnt=a.sum(0,dtype=np.int32)
        net=2*a[wrong[ids]].sum(0,dtype=np.int32)-cnt
        j=choose(cnt,net,len(ids))
        forced=int((net*(len(ids)+1)-cnt).argmax())
        if j>=0:
            oracle[ids]=A[ids,j]
        cells[str(cell)]=dict(rows=len(ids),errors=int(wrong[ids].sum()),choice='keep' if j<0 else names[j],
                             best_forced_rule=names[forced],best_forced_net=int(net[forced]))
        for source in np.unique(part[ids]):
            local=ids[part[ids]==source]
            a=A[local]
            cnt=a.sum(0,dtype=np.int32)
            net=2*a[wrong[local]].sum(0,dtype=np.int32)-cnt
            # Each focal passenger is a distinct booking. Subtract all of that row's
            # contributions to exactly recover a fit on only the other eligible orders.
            for i in local:
                x=A[i]
                j=choose(cnt-x,net-signed[i]*x,len(local)-1)
                if j>=0:
                    held[i]=x[j]
    return dict(cells=cells,hindsight=metrics(oracle,wrong,fold),other_orders=metrics(held,wrong,fold))


def run():
    raw,y,g,f,parent,blocks,flags,wrong,masks=context()
    ids=np.flatnonzero(masks['full'])
    assert len(ids)==2448 and wrong[ids].sum()==524
    with np.load(ROOT/'artifacts/nov26_tabicl_consensus_review/full.npz') as z:
        original=z['names'].tolist()+NEW
    archive=list(archives(raw,y,g,f,True)[0])
    groups={'original_99':original,'other_archives':archive,'supplementary':FAMILIES,'tabstar':['TabSTAR_finetuned'],
            'bill_density':['conditional_bills'],'tabnet_recipes':MODELS}
    assert list(groups)==list(blocks) and all(len(groups[k])==v.shape[1] for k,v in blocks.items())
    complete=np.column_stack([v[ids] for v in blocks.values()])
    names=[family+'/'+n for family,ns in groups.items() for n in ns]
    assert len(names)==len(set(names))==complete.shape[1]==127 and complete.any(1).all()
    part=exposure_partitions(raw,y,g,f,ids)
    side=np.where(parent[ids],'positive','negative')
    origin=raw.HomePlanet.iloc[ids].astype('string').fillna('missing').to_numpy(str)
    cryo=raw.CryoSleep.iloc[ids].astype('string').fillna('missing').to_numpy(str)
    cells={'all':np.repeat('all',len(ids)), 'direction':side,
           'direction_origin':np.char.add(np.char.add(side,'/'),origin)}
    cells['direction_origin_cryo']=np.char.add(np.char.add(cells['direction_origin'],'/'),cryo)
    sizes=np.unique(part,return_counts=True)[1]
    report=dict(rows=len(ids),errors=int(wrong[ids].sum()),correct=int((~wrong[ids]).sum()),
                holdout_method='Leave one solo order out inside the same base fold, reviewer query batch and EXAONE support regime. Select the best observed net rule on the other in-scope bookings of that same covariate cell. Allow keeping the baseline and prefer fewer flips on ties. No new search on the focal outcome.',
                scope='All 127 archived opinion columns are compared individually. The 109-view analysis removes the first 18 pre-averaged family representatives. A pair requires two distinct columns to oppose the baseline simultaneously. Related source columns are not independent replications.',
                validation_groups=len(sizes),validation_group_size_range=[int(sizes.min()),int(sizes.max())],
                versions={})
    for tag,start in [('127_archived_opinions',0),('109_without_family_means',18)]:
        a=complete[:,start:]
        ns=names[start:]
        i,j=np.triu_indices(len(ns),1)
        both=a[:,i]&a[:,j]
        report['versions'][tag]={}
        for k,m,rules in [('single',a,ns),('pair',both,[ns[x]+' & '+ns[v] for x,v in zip(i,j,strict=True)])]:
            out={scheme:scenario(m,rules,keys,part,wrong[ids],f[ids]) for scheme,keys in cells.items()}
            report['versions'][tag][k]=dict(number_of_rules=len(rules),partitions=out)
            print(tag,k,[(n,entry['hindsight']['net'],entry['other_orders']['net']) for n,entry in out.items()],flush=True)
    # Without EXAONE, training and query can span its four source regimes. Rows
    # entering the population solely through EXAONE have zero eligible actions
    # for every retained rule and contribute no evidence to rule choice.
    ix=np.array([i for i,name in enumerate(names) if i>=18 and name!='other_archives/EXAONE_grouped'])
    assert len(ix)==108
    broad=part//4
    cnt=np.unique(broad,return_counts=True)[1]
    a=complete[:,ix]
    ns=[names[i] for i in ix]
    i,j=np.triu_indices(len(ix),1)
    sensitivity={}
    for mode,table,rules in [('single',a,ns),('pair',a[:,i]&a[:,j],[ns[l]+' & '+ns[r] for l,r in zip(i,j,strict=True)])]:
        sensitivity[mode]={k:scenario(table,rules,labels,broad,wrong[ids],f[ids]) for k,labels in cells.items()}
        print('108',mode,[(k,v['other_orders']['net']) for k,v in sensitivity[mode].items()],flush=True)
    report['larger_calibration_without_exaone']=dict(partitions=len(cnt),size_range=[int(cnt.min()),int(cnt.max())],results=sensitivity)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
