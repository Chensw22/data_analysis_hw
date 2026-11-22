"""Benchmark the same broad release mistakes across archived predictions and honest combinations."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_partition_evidence_n23 import EXTRA, SOURCES, SEEDS
from broad_release_transfer_n23 import write, REPORT
from released_correction_audit_n23 import development


def summarize(hit, base_hit, masks):
    stable = hit.all(0)
    gained, lost = hit & ~base_hit, base_hit & ~hit
    return dict(accuracy=hit.mean(1).tolist(), fixed=gained.sum(1).tolist(), harmed=lost.sum(1).tolist(),
                stable={k:int((stable&m).sum()) for k,m in masks.items()})


def main():
    row,y,base,p,_,names,_ = panel()
    ids,targets,frame,groups,_ = load()
    np.testing.assert_array_equal(ids,row)
    np.testing.assert_array_equal(targets,y)
    targets,release,_ = development()
    np.testing.assert_array_equal(targets,y)
    extra = {**EXTRA, **dict(list(SOURCES.items())[:4])}
    extras,folds = [],[]
    for seed in SEEDS:
        _,values,fold = predictions(row,y,seed,extra)
        extras.append(values)
        folds.append(fold)
    p = np.concatenate([p,np.stack(extras)],axis=2)
    names = ["archive/" + k for k in names] + ["additional/" + k for k in extra]
    assert len(names)==len(set(names))==p.shape[2]
    base_hit = (base>=.5)==y
    hit = (p>=.5)==y[None,:,None]
    missed = np.logical_and.reduce([v!=y for v in release.values()])
    state = strata(frame)
    paid = missed & (state==2) & y
    zero = missed & (state==0) & ~y
    masks = dict(all_release_errors=missed, paid_transported=paid, zero_stayed=zero, rest=missed&~(paid|zero))
    stats = {name:summarize(hit[:,:,i],base_hit,masks) for i,name in enumerate(names)}
    statistics = {}
    stable = hit.all(0)
    for k,mask in masks.items():
        counts = stable[mask].sum(1)
        pairs = stable[mask].astype(int)
        freq = pairs.sum(0)
        union = freq[:,None]+freq[None,:]-pairs.T@pairs
        i,j = np.unravel_index(union.argmax(),union.shape)
        statistics[k] = dict(rows=int(mask.sum()), any_stable=int((counts>0).sum()),
                             none_ever=int((mask&~hit.any((0,2))).sum()),
                             best_pair=[names[i],names[j]],best_pair_union=int(union[i,j]),
                             largest_single=int(freq.max()))
    # Learn full-population mixtures solely on bookings disjoint from each query.
    # Only the observed state of the bills can enter the optional interaction.
    methods = {k:[] for k in ("global","bill_interactions")}
    for s,fold in enumerate(folds):
        features = np.column_stack([logit(base[s].clip(.005,.995)), logit(p[s].clip(.005,.995))])
        x = features / np.sqrt(features.shape[1])
        designs = dict(global_features=x, bill_interactions=np.c_[x,x*(state==0)[:,None],x*(state==2)[:,None]])
        for key,design in designs.items():
            votes = np.zeros(len(y),bool)
            for f in np.unique(fold):
                ids = np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=groups[ids]):
                    t,q = ids[a],ids[b]
                    assert not np.intersect1d(groups[t],groups[q]).size
                    model = LogisticRegression(C=1.,max_iter=500).fit(design[t],y[t])
                    votes[q] = model.predict(design[q])
            methods["global" if key=="global_features" else key].append(votes==y)
    blends = {k:summarize(np.stack(v),base_hit,masks) for k,v in methods.items()}
    write(REPORT/"release_miss_experts.json",dict(families=len(names),cohorts=statistics,
          members=stats,baseline=summarize(base_hit,base_hit,masks),blends=blends))
    print(len(names),statistics,blends)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
