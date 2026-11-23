"""Train manifest representations on outcome agreement across different bookings."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel
from broad_error_scope_balance_n23 import archived_votes
from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_refit_dissent_n23 import features
from broad_release_miss_coverage_n23 import summarize
from broad_booking_policy_replay_n23 import interval

REPORT = ROOT / "reports/nov23_population_supervision"


def contrast(z, labels, group, context):
    norm = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1.e-8)
    v = z / norm
    allowed = (group[:, None] != group) & (context[:, None] == context)
    pos = allowed & (labels[:, None] == labels)
    count = pos.sum(1, keepdims=True)
    active = (count.ravel() > 0) & (pos.sum(1) < allowed.sum(1))
    sim = (v @ v.T) / .25
    mass = np.exp(sim - 4.) * allowed
    prob = mass / np.maximum(mass.sum(1, keepdims=True), 1.e-12)
    desired = pos / np.maximum(count, 1)
    w = (prob - desired) * active[:, None] / max(active.sum(), 1)
    grad = ((w + w.T) @ v) / .25
    grad = (grad - (grad*v).sum(1, keepdims=True)*v) / norm
    loss = -np.sum(desired * np.log(np.maximum(prob, 1.e-15)) * active[:,None])/max(active.sum(),1)
    return loss, grad


def train(x, y, groups, cells, mode, t, q):
    scaler = StandardScaler().fit(x[t])
    design = scaler.transform(x).clip(-6, 6)
    a = np.c_[design, np.ones(len(design))]
    rng = np.random.default_rng(497)
    n = a.shape[1]
    p = [rng.normal(0, 1/np.sqrt(n), (n, 48)),rng.normal(0, 1/np.sqrt(49), (49,16)),
         rng.normal(0, 1/np.sqrt(17), 17)]
    first = [np.zeros_like(w) for w in p]
    second = [np.zeros_like(w) for w in p]
    pair_y = y.copy()
    if mode == "permuted":
        sham = np.random.default_rng(801)
        for cell in np.unique(cells[t]):
            ids=t[cells[t]==cell]
            pair_y[ids] = sham.permutation(pair_y[ids])
    context = np.zeros_like(cells) if mode == "all_pairs" else cells
    step = 0
    def forward(ix):
        h=np.tanh(a[ix]@p[0])
        hb=np.c_[h, np.ones(len(h))]
        z=hb@p[1]
        zb=np.c_[z, np.ones(len(z))]
        return h,hb,z,zb,expit(zb@p[2])
    for epoch in range(32):
        shuffled=rng.permutation(t)
        for ids in np.array_split(shuffled, int(np.ceil(len(t)/160))):
            h,hb,z,zb,prob=forward(ids)
            delta=(prob-y[ids])/len(ids)
            dz=delta[:,None]*p[2][None,:-1]
            if mode!="binary":
                dz += .25*contrast(z,pair_y[ids],groups[ids],context[ids])[1]
            gradients=[a[ids].T @ ((dz@p[1][:-1].T)*(1-h*h)),hb.T@dz,zb.T@delta]
            step+=1
            for j,g in enumerate(gradients):
                g=g+.0002*p[j]
                first[j]=.9*first[j]+.1*g
                second[j]=.999*second[j]+.001*g*g
                p[j]-=.002*(first[j]/(1-.9**step))/(np.sqrt(second[j]/(1-.999**step))+1.e-8)
    return forward(q)[-1]


def main():
    rows,y,frame,g,_=load()
    other,labels,masks=historical()
    np.testing.assert_array_equal(rows,other)
    np.testing.assert_array_equal(y,labels)
    x=features(frame)[0]
    _,cryo=np.unique(frame.CryoSleep,return_inverse=True)
    cells=cryo*3+strata(frame)
    hist,refs={},[]
    dest=ART/"nov23_population_supervision"
    dest.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            for key,v in (("row",rows),("y",y)): np.testing.assert_array_equal(z[key],v)
            fold,base=z["fold"],z["base"]
        out={}
        for mode in ("binary","all_pairs","matched","permuted"):
            pred=np.full(len(y),np.nan)
            for f in np.unique(fold):
                t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(g[t],g[q]).size
                pred[q]=train(x,y,g,cells,mode,t,q)
            out[mode]=pred
            comb=np.full(len(y),np.nan)
            d=logit(np.c_[base,pred].clip(.001,.999))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    comb[q]=LogisticRegression(C=.5).fit(d[t],y[t]).predict_proba(d[q])[:,1]
            out["combined_"+mode]=comb
        np.savez_compressed(dest/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**out)
        refs.append((base>=.5)==y)
        for k,v in out.items():
            assert np.isfinite(v).all()
            hist.setdefault(k,[]).append((v>=.5)==y)
        print(seed,[(k,round(h[-1].mean(),4)) for k,h in hist.items()],flush=True)
    results={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}
    cmp={}
    for k in ("matched","combined_matched"):
        for alt in ("binary","permuted","all_pairs"):
            b=k.replace("matched",alt)
            delta=np.array(hist[k],float)-np.array(hist[b],float)
            cmp[k+"_vs_"+b]={"mean_pp":100*delta.mean(),"interval_pp":interval(delta.mean(0),g)}
    write(REPORT/"results.json",dict(models=results,comparison=cmp))
    coverage()


def coverage():
    rows,y,base,archive,*_=panel()
    old=(archived_votes(rows,y,base,archive)==y[None,:,None])
    repeat=np.all((base>=.5)!=y,axis=0)
    old_stable=old.all(0).any(1)
    old_never=~old.any((0,2))
    assert int((repeat & old_stable).sum())==594
    history={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_population_supervision/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],rows)
            np.testing.assert_array_equal(z["y"],y)
            for key in ("binary","all_pairs","matched","permuted","combined_matched"):
                history.setdefault(key,[]).append((z[key]>=.5)==y)
    output={}
    for key,h in history.items():
        stable=np.logical_and.reduce(h)&repeat
        output[key]=dict(stable=int(stable.sum()), beyond_archive=int((stable&~old_stable).sum()),
                         beyond_ever_correct=int((stable&old_never).sum()))
    write(REPORT/"coverage.json",output)


if __name__=="__main__":
    with threadpool_limits(1):
        main()
