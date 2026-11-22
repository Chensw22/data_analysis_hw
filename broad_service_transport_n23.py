"""Compare distributions of passenger spending by the cost of redistributing among services."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, pdist
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_weak_disagreement_n23 import ART, ROOT, SEEDS, archive
from exaone_validation_data import CATEGORICAL
from relational_error_corrector import write

OUT = ART / "nov23_broad_service_transport"
REPORT = ROOT / "reports/nov23_broad_service_transport"


def service_tree(spending, y, train):
    # Each service is described by its label frequency at five positive amount
    # ranges, separately for bills with a single versus several active services.
    total = spending.sum(axis=1)
    multiple = (spending > 0).sum(1) > 1
    cuts = np.quantile(spending[train][spending[train] > 0], [.2, .4, .6, .8])
    signatures = []
    for service in spending.T:
        bands = np.digitize(service, cuts) + 5*multiple
        ix = train[service[train] > 0]
        denominator = np.bincount(bands[ix], minlength=10)
        numer = np.bincount(bands[ix], weights=y[ix], minlength=10)
        signatures.append((numer+12*y[train].mean()) / (denominator+12))
    tree = linkage(pdist(signatures), method="average")
    parts = np.eye(5).tolist()
    heights = [0.] * 5
    allocations, lengths = [], []
    for left, right, height, _ in tree:
        parent = np.zeros(5)
        for child in (int(left), int(right)):
            parent += parts[child]
            allocations.append(parts[child])
            lengths.append(height - heights[child])
        parts.append(parent.tolist())
        heights.append(height)
    # Root-to-leaf distances are equal. Normalization gives every topology the
    # same maximum possible distance between two unit spending portfolios.
    weights = np.asarray(lengths) / (2*heights[-1])
    return np.asarray(allocations).T * weights


def member_predictions(frame, y, train, query):
    spend = frame[BILLS].fillna(0).to_numpy(float)
    total = spend.sum(1)
    mass = np.divide(spend, total[:,None], out=np.zeros_like(spend), where=total[:,None]>0)
    design = frame.drop(columns=BILLS).copy()
    design["log_budget"] = np.log1p(total)
    design["n_active"] = (spend>0).sum(1)
    design["n_missing"] = frame[BILLS].isna().sum(axis=1)
    categories = design[CATEGORICAL].to_numpy()
    numeric = design.drop(columns=CATEGORICAL).to_numpy(float)
    missing = ~np.isfinite(numeric)
    numeric = np.where(missing, np.nanmedian(numeric[train],axis=0), numeric)
    low, high = np.quantile(numeric[train], [.02,.98], axis=0)
    numeric = ((numeric-low)/(high-low).clip(.1)).clip(0,1)
    rng = np.random.default_rng(225)
    anchor = rng.choice(train, min(450,len(train)), replace=False)
    rows = np.r_[train,query]
    context = cdist(numeric[rows], numeric[anchor], "cityblock")
    context += .4 * cdist(missing[rows].astype(float), missing[anchor].astype(float), "cityblock")
    for j in range(categories.shape[1]):
        context += (categories[rows,j,None] != categories[anchor,j])
    context = np.exp(-.55*context)
    geometry = service_tree(spend, y, train)
    geometries = dict(star=np.eye(5)/2, learned=geometry, shuffled=geometry[[2,4,0,1,3]])
    predictions = {}
    for name, transform in geometries.items():
        # Exact tree-Wasserstein distance is the edge-length-weighted L1 distance
        # of the proportions below every edge for equal-mass portfolios.
        emb = mass@transform
        transport = cdist(emb[rows], emb[anchor], "cityblock")
        # No positive observed spend has no defined portfolio; only compare its
        # composition as equal to another empty portfolio.
        one_empty = (total[rows,None]==0) != (total[anchor]==0)
        transport[one_empty] = 1.
        proximity = np.exp(-2*transport)
        x = np.column_stack([context, proximity*context, proximity])
        model = LogisticRegression(C=.25, max_iter=600).fit(x[:len(train)], y[train])
        predictions[name] = model.predict_proba(x[len(train):])[:,1]
    return predictions


def run():
    row,y,frame,groups,_ = load()
    ids,labels,baseline,eligible,masks,_,_ = archive()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    OUT.mkdir(exist_ok=True)
    plans = []
    for i,seed in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            np.testing.assert_array_equal(ref["row"],row)
            np.testing.assert_array_equal(ref["y"],y)
            np.testing.assert_array_equal(ref["base"],baseline[i])
            plans.append((str(seed),ref["fold"],baseline[i],None))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as ref:
        np.testing.assert_array_equal(ref["row"],row)
        np.testing.assert_array_equal(ref["y"],y)
        plans.append(("v131",ref["fold"],ref["v108"],ref["v131"]))
    report, hits = {}, {}
    for name,fold,base,inc in plans:
        bank = {}
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            opinions=member_predictions(frame,y,t,q)
            for key,prob in opinions.items():
                bank.setdefault(key,np.full(len(y),np.nan))[q]=prob
                x=logit(np.column_stack([base[q],prob]).clip(.002,.998))
                for a,b in GroupKFold(3,shuffle=True,random_state=513).split(q,groups=groups[q]):
                    assert not np.intersect1d(groups[q[a]],groups[q[b]]).size
                    est=LogisticRegression(C=.15).fit(x[a],y[q[a]])
                    bank.setdefault(key+"_combined",np.full(len(y),np.nan))[q[b]]=est.predict_proba(x[b])[:,1]
            print(name,k,flush=True)
        reference=base>=.5 if inc is None else inc
        before=reference==y
        report[name]={}
        for key,p in bank.items():
            assert np.isfinite(p).all()
            if inc is not None and key.endswith("_combined"):
                p=np.where(inc!=(base>=.5),inc,p)
            hit=(p>=.5)==y
            gain=hit.astype(int)-before
            errscore=np.where(reference,1-p,p)
            report[name][key]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                by_fold=[int(gain[fold==k].sum()) for k in np.unique(fold)],
                correct_by_scope={s:int((hit&m).sum()) for s,m in masks.items()},
                mirror_auc=float(roc_auc_score(masks["weak409"][eligible],errscore[eligible])) if inc is None else None)
            if inc is None:
                hits.setdefault(key,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,**bank)
    old=(baseline>=.5)==y
    stable={}
    for key,arr in hits.items():
        good=np.stack(arr)
        stable[key]=dict(correct_each_time={s:int((good.all(0)&m).sum()) for s,m in masks.items()},
            ever_correct={s:int((good.any(0)&m).sum()) for s,m in masks.items()},
            paired_gain95_pp=interval((good.astype(float)-old).mean(0),groups))
    write(REPORT/"result.json",dict(bills=BILLS,plans=report,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
