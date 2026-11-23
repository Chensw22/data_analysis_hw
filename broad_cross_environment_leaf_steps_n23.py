"""Constrain each boosting update by the signs supported in multiple booking environments."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import CATEGORICAL, assign, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write


REPORT = ROOT/'reports/nov23_cross_environment_leaf_steps'
OUT = ART/'nov23_cross_environment_leaf_steps'
TREE_COUNT = 360
DEPTH = 5
RATE = .07


def centered(grad, curvature):
    """One Newton intercept and a regularized, centered update per leaf."""
    weight = curvature.sum(axis=-1, keepdims=True)
    root = grad.sum(axis=-1, keepdims=True)/weight.clip(.01)
    contrast = (grad-curvature*root)/(curvature+3.)
    return root, contrast


def coefficients(leaf, target, logits, environments):
    nleaf = 2**DEPTH
    prob = expit(logits)
    grad, curvature = target-prob, prob*(1-prob)
    dg = np.bincount(leaf, weights=grad, minlength=nleaf)
    dh = np.bincount(leaf, weights=curvature, minlength=nleaf)
    root, pooled = centered(dg, dh)
    cap = abs(pooled).copy()
    for env in environments:
        count = int(np.max(env))+1
        cell = env*nleaf+leaf
        local_g = np.bincount(cell, weights=grad, minlength=count*nleaf).reshape(count,nleaf)
        local_h = np.bincount(cell, weights=curvature, minlength=count*nleaf).reshape(count,nleaf)
        _, local = centered(local_g,local_h)
        # Limit a signed update only when a sufficiently precise environment estimate
        # contradicts its full magnitude. Tiny/empty groups supply no practical cap.
        upper = np.sign(pooled)[None,:]*local+1.28/np.sqrt(local_h.clip(.001))
        cap = np.minimum(cap, np.maximum(0, upper.min(0)))
    used = np.sign(pooled)*cap
    return (root+used).ravel(), np.array([dh@abs(pooled), dh@(abs(pooled)-cap),
                                          (dh*(cap+1e-12<abs(pooled))).sum(), dh.sum()])


def fit(frame, g, region, y, t, q):
    assert not np.intersect1d(g[t],g[q]).size
    tree = CatBoostClassifier(iterations=TREE_COUNT,learning_rate=RATE,depth=DEPTH,
                              thread_count=2,allow_writing_files=False,verbose=False,random_seed=811)
    tree.fit(frame.iloc[t], y[t], cat_features=CATEGORICAL)
    leaves = tree.calc_leaf_indexes(frame.iloc[np.r_[t,q]]).astype(int)
    assert leaves.shape==(len(t)+len(q),TREE_COUNT)
    regimes={k:assign(t,g,region,k,4,811) for k in ['cabin','order']}
    random=[assign(t,g,region,'random',4,seed) for seed in [211,813]]
    cases={'pooled':[], 'random':random, 'cabin':[regimes['cabin']],
           'order':[regimes['order']], 'both':list(regimes.values())}
    pred={'catboost':tree.predict_proba(frame.iloc[q])[:,1]}
    measure={}
    for name,envs in cases.items():
        logits=np.full(len(leaves),logit(y[t].mean()))
        sums=np.zeros(4)
        for leaf in leaves.T:
            v,info=coefficients(leaf[:len(t)], y[t], logits[:len(t)],envs)
            logits+=RATE*v[leaf]
            sums+=info
        pred[name]=expit(logits[len(t):])
        measure[name]=dict(removed_magnitude=float(sums[1]/sums[0]),constrained_mass=float(sums[2]/sums[3]))
    return pred,measure


def main():
    row,y,frame,g,region=load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        history=z['votes']
    wrong=(history!=y[None,:,None]).all((0,2))
    right=(history==y[None,:,None]).all((0,2))
    assert wrong.sum()==983 and right.sum()==5343
    OUT.mkdir(exist_ok=True)
    plans,hits,measures={},{},{}
    for seed in [*map(str,SEEDS),'v131']:
        folds,ref,opinions,_=read_opinions(seed,row,y)
        scores={}
        measures[seed]=[]
        for f in np.unique(folds):
            q=np.flatnonzero(folds==f)
            t=np.flatnonzero((folds!=f)&~np.isin(g,g[q]))
            pred,effects=fit(frame,g,region,y,t,q)
            measures[seed].append(effects)
            for name,p in pred.items():
                scores.setdefault(name,np.full(len(y),np.nan))[q]=p
        assert all(np.isfinite(v).all() for v in scores.values())
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=folds,ref=ref,base=opinions[:,0],**scores)
        votes={k:p>=.5 for k,p in scores.items()}
        for k in ['random','cabin','order','both']:
            adjustment=logit(scores[k])-logit(scores['pooled'])
            votes['transfer_'+k]=np.where(ref==(opinions[:,0]>=.5),
                logit(opinions[:,0].clip(.002,.998))+adjustment>=0,ref)
        votes['reference']=ref
        correct={k:v==y for k,v in votes.items()}
        plans[seed]={}
        for key,h in correct.items():
            compare=correct['reference'] if key.startswith('transfer_') else correct['pooled']
            gain=h.astype(float)-compare
            plans[seed][key]=dict(vs_reference=paired(h,correct['reference']),matched=paired(h,compare),
                matched_fold_delta=[int(gain[folds==f].sum()) for f in np.unique(folds)],
                corrected_joint_wrong=int((h&wrong).sum()),correct_joint_right=int((h&right).sum()))
            if seed!='v131':
                hits.setdefault(key,[]).append(h)
        print(seed, plans[seed]['both']['matched'],plans[seed]['transfer_both']['matched'],flush=True)
    panel={k:np.stack(v) for k,v in hits.items()}
    miss=(~panel['reference']).sum(0)
    analysis={}
    for key,h in panel.items():
        compare=panel['reference'] if key.startswith('transfer_') else panel['pooled']
        delta=(h.astype(float)-compare).mean(0)
        analysis[key]=dict(matched_gain_pp=float(100*delta.mean()),paired_ci95_pp=interval(delta,g),
            stable_recovered=int(h[:,wrong].all(0).sum()),stable_retained=int(h[:,right].all(0).sum()),
            stable_wrong_right_group=int((~h[:,right]).all(0).sum()),
            stable_by_times_wrong=[int(h[:,miss==n].all(0).sum()) for n in range(4)])
    write(REPORT/'result.json',dict(plans=plans,grouped=analysis,update_effects=measures))


if __name__=='__main__':
    with threadpool_limits(2):main()
