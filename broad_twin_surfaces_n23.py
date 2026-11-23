"""Learn separate nonlinear surfaces near each outcome and test their transfer."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_passenger_transport_n23 import manifest_coordinates
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_twin_surfaces"


def basis(x, t):
    # A fixed landmark draw from training rows supplies a common finite
    # feature space for the single-margin and two-surface predictors.
    rng = np.random.default_rng(421)
    centers = x[rng.choice(t, 256, replace=False)]
    d = cdist(x, centers, "sqeuclidean")
    width = max(np.quantile(d[t], .20), 1.e-6)
    gram = np.exp(-cdist(centers, centers, "sqeuclidean") / width)
    lam, vec = np.linalg.eigh(gram)
    valid = lam > 1.e-6
    features = np.exp(-d / width) @ (vec[:, valid] / np.sqrt(lam[valid]))
    return np.column_stack([features, x, np.ones(len(x))])


def surface(x, y, kind):
    target = 2*y.astype(float)-1
    def cost(theta):
        z = x @ theta
        if kind == "parallel":
            short = np.maximum(1-target*z, 0.)
            loss, dz = short@short, -2*target*short
        else:
            own = y == (kind == "positive")
            short = np.maximum(1-z[~own], 0.)
            loss = z[own]@z[own] + short@short
            dz = 2*z.copy()
            dz[~own] = -2*short
        grad = x.T@dz
        grad[:-1] += 2*theta[:-1]
        return loss + theta[:-1]@theta[:-1], grad
    sol = minimize(cost, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter":800,"ftol":1.e-10})
    assert sol.success, sol.message
    return sol.x


def run_seed(x, y, group, fold, base):
    margin = np.zeros(len(y))
    signed = np.zeros((len(y),2))
    for f in np.unique(fold):
        t, q = np.flatnonzero(fold!=f), np.flatnonzero(fold==f)
        assert not np.intersect1d(group[t],group[q]).size
        phi = basis(x,t)
        for k,name in enumerate(("negative","positive","parallel")):
            coef = surface(phi[t],y[t],name)
            raw = phi[q]@coef
            if name=="parallel":
                margin[q] = raw
            else:
                signed[q,k] = raw/np.linalg.norm(coef[:-1])
    distance = np.abs(signed)
    ratio = np.log((distance[:,0]+1.e-5)/(distance[:,1]+1.e-5))
    outputs = {"single":expit(margin), "twin":expit(ratio)}
    # Both signed distances can help distinguish a typical member from a
    # passenger far beyond the fitted surface. The distance rule uses only
    # the ratio. No retrospective error identities are supplied.
    b = logit(base.clip(.001,.999))
    for f in np.unique(fold):
        ids = np.flatnonzero(fold==f)
        for a,c in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=group[ids]):
            t,q=ids[a],ids[c]
            assert not np.intersect1d(group[t],group[q]).size
            main = np.c_[b, margin]
            sham = signed.copy()
            rng = np.random.default_rng(428)
            for split in (t,q):
                bands = np.digitize(b[split],np.quantile(b[t],np.arange(1,10)/10))
                for band in np.unique(bands):
                    cell = split[bands==band]
                    sham[cell] = signed[rng.permutation(cell)]
            for name, a2 in (("cal_single",main),("cal_ratio",np.c_[main,ratio]),
                             ("cal_twin",np.c_[main,signed,distance]),
                             ("shuffled",np.c_[main,sham,np.abs(sham)])):
                means,sd = a2[t].mean(0),a2[t].std(0).clip(.001)
                model=LogisticRegression(C=.3,max_iter=400).fit((a2[t]-means)/sd,y[t])
                outputs.setdefault(name,np.full(len(y),np.nan))[q]=model.predict_proba((a2[q]-means)/sd)[:,1]
    return outputs, signed


def scopes(y, frame):
    labels,releases,m=development()
    np.testing.assert_array_equal(labels,y)
    common=np.logical_and.reduce([v!=y for v in releases.values()])
    bill=strata(frame)
    core=common&(((bill==2)&y)|((bill==0)&~y))
    return dict(ever=m["ever"],repeat=m["repeat"],common=common,core=core,
                repeat_outside=m["repeat"]&~core)


def main():
    row,y,frame,g,_=load()
    x=manifest_coordinates(frame)
    masks=scopes(y,frame)
    dest=ART/"nov23_twin_surfaces"
    dest.mkdir(exist_ok=True)
    history,refs={},[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as cache:
            for k,v in (("row",row),("y",y)): np.testing.assert_array_equal(cache[k],v)
            base,fold=cache["base"],cache["fold"]
        out,signed=run_seed(x,y,g,fold,base)
        np.savez_compressed(dest/f"{seed}.npz",row=row,y=y,fold=fold,base=base,signed=signed,**out)
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in out.items():
            assert np.isfinite(p).all()
            h=(p>=.5)==y
            history.setdefault(k,[]).append(h)
        print(seed,[(k, int((hs[-1]&~ref).sum()), int((~hs[-1]&ref).sum())) for k,hs in history.items()],flush=True)
    write(REPORT/"twin.json",{k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()})


if __name__=="__main__":
    with threadpool_limits(2):
        main()
