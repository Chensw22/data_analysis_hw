"""Cross-fit conditional rank patterns of outer-fold experts across every passenger."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, logit, ndtri
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, predictions
from relational_error_corrector import write

# These are first-stage fits: all scores in a given outer validation fold use
# only other folds' outcomes. A meta-model trains strictly within that fold.
MEMBERS = {
    "registration": ("bill_matched_register","register"),
    "economic": ("bill_matched_register","economic"),
    "budget": ("simplex_likelihood","budget_prior"),
    "relations": ("passenger_relation","relation_both"),
    "field_interactions": ("field_factors","order3"),
    "bayesian_forest": ("bart_long","bart"),
    "aligned_network": ("shortcut_alignment","aligned"),
    "group_distribution": ("deployment_booking_mix","party_distribution"),
}


def conditional_ranks(base, opinions, train):
    # Measure how high each expert ranks a person relative to other people
    # given similar base risk. No labels are used to place these landmarks.
    order=np.argsort(base[train],kind="stable")
    blocks=np.array_split(train[order],8)
    edges=[(base[a[-1]]+base[b[0]])/2 for a,b in zip(blocks,blocks[1:])]
    bands=np.searchsorted(edges,base)
    ranks=np.empty_like(opinions)
    for i,ids in enumerate(blocks):
        mask=bands==i
        for j in range(opinions.shape[1]):
            sample=np.sort(opinions[ids,j])
            v=opinions[mask,j]
            left,right=np.searchsorted(sample,v,"left"),np.searchsorted(sample,v,"right")
            ranks[mask,j]=ndtri(((left+right)/2+.5)/(len(ids)+1))
    return ranks,bands


def fit_offset(train, query, x, base, y, penalty):
    z=logit(np.clip(base,.001,.999))
    a=x[train]
    def loss(w):
        value=z[train]+a@w
        return (np.logaddexp(0,value).sum()-value@y[train]+.5*(penalty*w)@w,
                a.T@(expit(value)-y[train])+penalty*w)
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":400})
    if not fit.success:
        raise RuntimeError(fit.message)
    return expit(z[query]+x[query]@fit.x)


def design(base, ranks, train, rng):
    z=logit(base.clip(.001,.999))
    prior=np.column_stack([np.ones(len(base)),z,np.maximum(z-1.,0),np.maximum(-z-1.,0)])
    linear=np.column_stack([prior,ranks,ranks*(base[:,None]-.5)])
    # Landmarks model multi-expert combinations, including the shape of
    # disagreements when every expert casts the same binary vote.
    landmarks=ranks[rng.choice(train,96,replace=False)]
    dist=cdist(ranks,landmarks,"sqeuclidean")
    scale=max(float(np.median(dist[train])),.1)
    local=np.exp(-2*dist/scale)
    return dict(base_risk=prior,relative_levels=linear,relative_patterns=np.column_stack([linear,local]))


def run():
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    with np.load(ART/"nov23_archive_recovery/masks.npz") as masks:
        np.testing.assert_array_equal(masks["row"],row)
        hard=masks["originally_unresolved"]
        deepest=masks["still_unresolved"]
    g=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()[row]
    folder=ART/"nov23_disagreement_patterns"
    folder.mkdir(exist_ok=True)
    report={}
    for seed in SEEDS:
        base,opinions,fold=predictions(row,y,seed,MEMBERS)
        results={}
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=311).split(outer,groups=g[outer]):
                t,q=outer[a],outer[b]
                assert not np.intersect1d(g[t],g[q]).size
                ranks,bands=conditional_ranks(base,opinions,t)
                shuffled=ranks.copy()
                for part in (t,q):
                    for band in np.unique(bands[part]):
                        ix=part[bands[part]==band]
                        shuffled[ix]=ranks[np.random.default_rng(73+int(band)).permutation(ix)]
                variants=design(base,ranks,t,np.random.default_rng(600))
                variants["shuffled_patterns"]=design(base,shuffled,t,np.random.default_rng(600))["relative_patterns"]
                for name,x in variants.items():
                    penalty=np.full(x.shape[1],40.)
                    penalty[:4]=[5.,20.,25.,25.]
                    results.setdefault(name,np.full(len(y),np.nan))[q]=fit_offset(t,q,x,base,y,penalty)
        scores={}
        ref=(base>=.5)==y
        for name,s in results.items():
            assert np.isfinite(s).all()
            hit=(s>=.5)==y
            scores[name]=dict(accuracy=float(hit.mean()),fixed=int((hit&~ref).sum()),
                              harmed=int((~hit&ref).sum()),correct_on_365=int(hit[hard].sum()),
                              correct_on_252=int(hit[deepest].sum()),logloss=float(np.mean(-np.where(y,np.log(s),np.log1p(-s)))))
        np.savez_compressed(folder/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**results)
        report[seed]=scores
        print(seed,scores,flush=True)
    write(ROOT/"reports/nov23_disagreement_patterns/result.json",dict(members=MEMBERS,results=report))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
