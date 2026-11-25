"""Test whether actual attention to passenger fields improves decisions on TabNet countervotes."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import fit
from attentive_tabnet_n26 import OUT, MODELS, BILLS
from frozen_queue_shared_errors_n26 import paired
from target_mix_expert_routing_n26 import ROOT, batches, ticker


def signals(raw,y,g,f):
    probability=np.empty((len(y),len(MODELS)))
    features=None
    for k in np.unique(f):
        with np.load(OUT/f'fold{k}.npz') as z:
            t,q = z['t'],z['q']
            np.testing.assert_array_equal(t,np.flatnonzero(f!=k))
            np.testing.assert_array_equal(q,np.flatnonzero(f==k))
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
            np.testing.assert_array_equal(z['train_y'],y[t])
            assert not np.isin(g[q],g[t]).any()
            labels=z['names'].tolist()
            sets=[{'HomePlanet','Destination','VIP'}, {'deck','berth','side'}, {'Age','order_size'},
                  {'CryoSleep'}, set(BILLS)|{'bill_total','paid_services'}]
            masks=[np.isin(labels,list(fields)) for fields in sets]
            masks.append(np.array([n.endswith('_missing') for n in labels]))
            assert (np.sum(masks,axis=0)==1).all()
            attention=[]
            for j,name in enumerate(MODELS):
                probability[q,j]=z[name]
                attrs=z[name+'_importance']
                attention.append(np.column_stack([attrs[:,mask].sum(1) for mask in masks]))
            a=np.stack(attention,axis=1)
            spread=np.std(a,axis=1)
            # Entropy and disagreement concern observed network masks, not a causal importance claim.
            entropy=-(a*np.log(a.clip(1.e-8,1))).sum(2)
            probdev=probability[q]-probability[q].mean(1,keepdims=True)
            packet=np.column_stack([a.mean(1),spread,entropy,(a*probdev[:,:,None]).sum(1)])
            if features is None:
                features=np.zeros((len(y),packet.shape[1]))
            features[q]=packet
    return probability,features


def displaced(a,fields,parent,average,seed):
    rng=np.random.default_rng(seed)
    # Keep predicted class and approximate TabNet confidence fixed while breaking passenger-to-mask alignment.
    bucket=parent.astype(int)*10+np.minimum(np.floor(5*average[a]).astype(int),4)
    move=np.arange(len(a))
    for key in np.unique(bucket):
        loc=np.flatnonzero(bucket==key)
        move[loc]=rng.permutation(loc)
    return fields[a[move]]


def run():
    raw,y,g,f,known,versions=paired()
    parent=versions[:,-1]
    common=known & (versions==parent[:,None]).all(1)
    with np.load(ROOT/'artifacts/nov26_independently_calibrated_release_risk/held_risk.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[z['ids']])
        high=np.zeros(len(y),bool)
        high[z['ids']]=z['top20']
    probs,attention=signals(raw,y,g,f)
    average=probs.mean(1)
    raw_scores=logit(probs.clip(.002,.998))
    decisions={}
    for b in batches(True):
        if b.tag!='full':
            continue
        t,q=b.train_ids,b.ids
        old=b.train_cell%2!=0
        now=parent[q]
        assert np.unique(f[np.r_[t,q]]).size==1 and not np.isin(g[t],g[q]).any()
        dissent=(average[t]>=.5)!=old
        challenge=(average[q]>=.5)!=now
        def design(ids,ref,ordinary):
            votes=raw_scores[ids]*np.where(ref,-1,1)[:,None]
            return np.column_stack([ordinary,votes,votes.std(1),votes.min(1)])
        xt,xq=design(t,old,b.train_fields),design(q,now,b.query_fields)
        w=np.where(dissent,1.,.2)
        flags={}
        for n,a,c in [('probability',None,None),('attention',attention[t],attention[q]),
                      ('displaced',displaced(t,attention,old,average,772),displaced(q,attention,now,average,494))]:
            train=xt if a is None else np.column_stack([xt,a])
            query=xq if c is None else np.column_stack([xq,c])
            risk=fit(train,b.train_error,query,w)
            flags[n]=challenge & (risk>.5)
        flags.update(added_by_attention=flags['attention'] & ~flags['probability'],
                     only_true_attention=flags['attention'] & ~flags['displaced'],
                     actual_and_scores=flags['attention'] & flags['probability'])
        for n,a in flags.items():
            decisions.setdefault(n,np.zeros(len(y),bool))[q]=a
    report={}
    for n,a in decisions.items():
        report[n]={tag:ticker(mask,y!=parent,a) for tag,mask in
                   [('all',known),('common',common),('high_common',high & common),('other_common',~high & common)]}
        report[n]['net_by_fold']=[int((np.where(y!=parent,1,-1)*a*common*(f==k)).sum()) for k in np.unique(f)]
        print(n,report[n]['common'],report[n]['high_common'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/tabnet_attention_reliability.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
