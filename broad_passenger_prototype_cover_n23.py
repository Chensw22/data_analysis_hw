"""Choose passenger exemplars for broad same-class coverage and measure their errors."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_dependence_anchors_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts, coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write

BUDGET = 64
NEIGHBORS = 96


def neighborhood(features,g,t):
    # A training group can have multiple passengers. They cannot endorse one another.
    extra = max(np.unique(g[t],return_counts=True)[1])
    distance, index = cKDTree(features[t]).query(features[t],k=NEIGHBORS+extra)
    other = g[t[index]]!=g[t,None]
    keep = other & (np.cumsum(other,axis=1)<=NEIGHBORS)
    assert (keep.sum(1)==NEIGHBORS).all()
    distance = distance[keep].reshape(len(t),NEIGHBORS)
    index = index[keep].reshape(len(t),NEIGHBORS)
    return index, np.maximum(distance[:,-1],.001), distance


def similarity(distance,radius):
    return np.maximum(1-(distance/radius)**2,0.)


def choose(index,weight,labels,cls,penalty):
    # Each additional prototype receives credit for improvements over the strongest
    # previous representative of this class, with an optional loss for opposite labels.
    candidates = np.flatnonzero(labels==cls)
    neighbors = index[candidates]
    strength = weight[candidates]
    reward = np.where(labels[neighbors]==cls,1.,-penalty)
    best = np.zeros(len(labels))
    selected = np.zeros(len(candidates),bool)
    sequence = []
    for _ in range(BUDGET):
        gain = (np.maximum(strength-best[neighbors],0.)*reward).sum(1)
        gain[selected] = -np.inf
        pos = np.argmax(gain)
        selected[pos] = True
        sequence.append(candidates[pos])
        ids = neighbors[pos]
        best[ids] = np.maximum(best[ids],strength[pos])
    return np.array(sequence)


def run_fold(frame,g,y,t,q):
    features,_ = coordinates(frame,t)
    neighbors,radius,distance = neighborhood(features,g,t)
    strength = similarity(distance,radius[:,None])
    rng = np.random.default_rng(1193)
    scores = {}
    for key,penalty in [('random',None),('coverage',0.),('discriminative',1.)]:
        affinities = []
        for cls in (False,True):
            pool = np.flatnonzero(y[t]==cls)
            support = rng.choice(pool,BUDGET,replace=False) if penalty is None else choose(neighbors,strength,y[t],cls,penalty)
            # Keep the same compact-support kernel and adaptive prototype radii used for selection.
            delta = features[q,None,:]-features[t[support]][None,:,:]
            proximity = similarity(np.sqrt(np.einsum('nkd,nkd->nk',delta,delta)),radius[support])
            affinities.append(proximity.max(1))
        negative,positive = affinities
        scores[key] = expit(np.log((y[t].sum()+1)/((~y[t]).sum()+1))+4*(positive-negative))
        scores[key+'_supported'] = np.maximum(positive,negative)>0
    return scores


def main():
    row,y,frame,groups,_ = load()
    wrong,right,hard = cohorts(row,y)
    plans,history = {},{}
    directory = ART/'nov23_passenger_prototype_cover'
    directory.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as data:
            np.testing.assert_array_equal(row,data['row'])
            np.testing.assert_array_equal(y,data['y'])
            fold,base,ref = data['fold'],data['base'],data['ref']
        bank = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            for key,p in run_fold(frame,groups,y,t,q).items():
                bank.setdefault(key,np.full(len(y),np.nan))[q] = p
        assert all(np.isfinite(p).all() for p in bank.values())
        hits = {'reference':ref==y}
        support_counts = {}
        for key in ['random','coverage','discriminative']:
            supported = bank[key+'_supported'].astype(bool)
            p = bank[key]
            support_counts[key] = int(supported.sum())
            active = supported & (ref==(base>=.5))
            hits[key] = np.where(active,p>=.5,ref)==y
            hits['quarter_'+key] = np.where(active,.75*base+.25*p>=.5,ref)==y
        metrics = {k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed] = dict(changes=metrics,support=support_counts)
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},support_counts,flush=True)
    baseline = np.stack(history['reference'])
    stable = {}
    for k,h in history.items():
        h = np.stack(h)
        stable[k] = dict(hard_stable=int((hard & h.all(0)).sum()),hard_ever=int((hard & h.any(0)).sum()),
                         common_stable=int((wrong & h.all(0)).sum()),stable_harms=int((right & ~h.any(0)).sum()),
                         gain95_pp=interval((h.astype(float)-baseline).mean(0),groups))
    write(REPORT/'prototype_cover.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
