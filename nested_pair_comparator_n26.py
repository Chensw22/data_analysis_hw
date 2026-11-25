"""Calibrate comparisons against real passengers with nearly identical teacher opinions."""
import json

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from full_roster_booking_validation_n23 import ROOT, data
from matched_remaining_solo_fields_n26 import matched, raw_fields
from scale_teacher_exception_training_n26 import CACHE
from solo_remaining_countervotes_n26 import context
from tabnet_attention_transfer_n26 import RESULT, collect
from target_mix_expert_routing_n26 import ticker

OUT = ROOT / 'artifacts/nov26_nested_pair_comparator'


def pair_design(x, a, other, b):
    return np.column_stack([x[a] - other[b], (x[a] + other[b]) / 2])


def models(x, a, b, seed):
    rows = np.vstack([pair_design(x,a,x,b), pair_design(x,b,x,a)])
    direction = np.random.default_rng(seed).integers(0,2,size=len(a)).astype(bool)
    result = []
    for first in (np.ones(len(a),bool), direction):
        model = HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=7,min_samples_leaf=20,
                                               l2_regularization=5,early_stopping=False,random_state=seed)
        model.fit(rows,np.r_[first,~first])
        result.append(model)
    return result


def compare(x, opinion, prior, bank_x, bank_p, bank_prior, judges):
    margins = np.empty((len(x),len(judges)))
    for side in (False,True):
        q, bank = np.flatnonzero(prior==side), np.flatnonzero(bank_prior==side)
        nearest = NearestNeighbors(n_neighbors=min(12,len(bank))).fit(bank_p[bank])
        neighbors = bank[nearest.kneighbors(opinion[q],return_distance=False)]
        a,b = np.repeat(q,neighbors.shape[1]),neighbors.ravel()
        forward = pair_design(x,a,bank_x,b)
        backward = pair_design(bank_x,b,x,a)
        for j,model in enumerate(judges):
            margin = (model.decision_function(forward) - model.decision_function(backward)) / 2
            margins[q,j] = margin.reshape(len(q),-1).mean(1)
    return margins


def score_fold(raw,y,g,t,q,k,z,fields):
    tri, panel, inner = z['soft'], z['outer_single'], z['inner_fold']
    direction_t = tri.mean(1)>=.5
    logits_t = logit(tri.clip(.003,.997))
    result = {}
    support = []
    for j in range(3):
        available = np.flatnonzero(inner==j)
        # Both the fitter and calibrator use predictions from the identical teacher fit,
        # which excluded all their labels and the outer validation orders.
        shard = partition(g[t[available]],'solo-pair-comparator:',3)
        train,cal = available[shard!=0],available[shard==0]
        assert not np.isin(g[t[train]],np.r_[g[t[cal]],g[q]]).any()
        assert not np.isin(g[t[inner!=j]],np.r_[g[t[available]],g[q]]).any()
        observed = y[t] != direction_t
        # Use one representative per training order to prevent paired passengers sharing an order.
        _, index = np.unique(g[t[train]],return_index=True)
        train=train[index]
        a,b = matched(train,direction_t,tri,observed)
        assert not np.isin(g[t[a]], g[t[b]]).any()
        judges=models(fields[t],a,b,764+k*3+j)
        logits_q=logit(panel[j].clip(.003,.997))
        prior_q=panel[j].mean(1)>=.5
        bank=fields[t[train]],tri[train],direction_t[train]
        mc=compare(fields[t[cal]],tri[cal],direction_t[cal],*bank,judges)
        mq=compare(fields[q],panel[j],prior_q,*bank,judges)
        # Learn the passenger's binary outcome (same direction for opposite priors)
        # from the calibrated teachers and the signed amount of error evidence.
        sign_t=np.where(direction_t[cal],-1,1)
        sign_q=np.where(prior_q,-1,1)
        for key,col in [('opinions',None),('matched',0),('randomized',1)]:
            xc,xq=logits_t[cal],logits_q
            if col is not None:
                xc=np.column_stack([xc,mc[:,col],mc[:,col]*sign_t])
                xq=np.column_stack([xq,mq[:,col],mq[:,col]*sign_q])
            model=LogisticRegression(C=.3,max_iter=250).fit(xc,y[t[cal]])
            result.setdefault(key,[]).append(model.predict_proba(xq)[:,1])
        support.append(len(a))
    return result,support


def run():
    raw,y,g,fold=data()
    blocks=raw_fields(raw,np.zeros(len(y),bool))
    # The second half duplicates the first when all reference labels are zero.
    fields=blocks['both'][:,:blocks['both'].shape[1]//2]
    scores={}
    support=[]
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(CACHE/f'fold{k}.npz') as z:
            for name,expected in [('t',t),('q',q),('train_y',y[t])]:
                np.testing.assert_array_equal(z[name],expected)
            out,counts=score_fold(raw,y,g,t,q,k,z,fields)
        for name,values in out.items():
            scores.setdefault(name,np.full((len(y),3),np.nan))[q]=np.stack(values,axis=1)
        support.append(counts)
        print('fold',k,counts,flush=True)
    assert all(np.isfinite(v).all() for v in scores.values())
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',ids=raw.PassengerId.to_numpy(str),fold=fold,**scores)
    _,truth,_,other_fold,parent,_,_,wrong,masks=context()
    np.testing.assert_array_equal(y,truth)
    np.testing.assert_array_equal(fold,other_fold)
    report={'training_pairs':support,'accuracy':{k:float(np.mean((v.mean(1)>=.5)==y)) for k,v in scores.items()}}
    cohorts={key:(mask,parent) for key,mask in masks.items()}
    for tag,items in collect()[0].items():
        mask=np.zeros(len(y),bool)
        ref=np.zeros(len(y),bool)
        for ids,common,base,actions in items:
            mask[ids]=common & ~actions['actual_and_scores']
            ref[ids]=base
        cohorts['all_common_'+tag]=mask,ref
    for tag,(scope,ref) in cohorts.items():
        edits={n:(v.mean(1)>=.5)!=ref for n,v in scores.items()}
        all_views=((scores['matched']>=.5)!=ref[:,None]).all(1)
        edits.update(unique_to_matched=edits['matched'] & ~edits['opinions'] & ~edits['randomized'],
                     required_for_change=edits['matched'] & ~edits['opinions'],
                     unanimous_new=all_views & ~edits['opinions'])
        report[tag]={}
        for n,act in edits.items():
            v=ticker(scope,y!=ref,act)
            delta=scope*act*np.where(y!=ref,1,-1)
            v['net_by_fold']=[int(delta[fold==k].sum()) for k in range(8)]
            report[tag][n]=v
        print(tag,report[tag]['unique_to_matched'],flush=True)
    (RESULT.parent/'nested_pair_comparator.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
