"""Probe whether independent booking outcomes can be recovered from neural training trajectories."""
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_partition_evidence_n23 import ART, ROOT, SEEDS, evaluate, historical, interval, load, write
from broad_weak_complement_n23 import combine
from class_analogue_experts_n23 import design

OUT = ART / "nov23_broad_neural_path"
REPORT = ROOT / "reports/nov23_broad_neural_path"
EPOCHS = (2, 8, 32, 80)


def trajectory(frame, y, t, q, seed):
    _, features = design(frame,t)
    features = np.column_stack([features, np.log(frame.BookingSize)])
    x = StandardScaler().fit(features[t]).transform(features).clip(-6,6).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(64,32), alpha=.2, batch_size=128,
                          learning_rate_init=.001, shuffle=False, random_state=seed)
    rng = np.random.default_rng(seed)
    columns = []
    for epoch in range(1, EPOCHS[-1]+1):
        a = rng.permutation(t)
        model.partial_fit(x[a], y[a], classes=[False,True])
        if epoch in EPOCHS:
            columns.append(model.predict_proba(x[q])[:,1])
    return np.column_stack(columns).clip(.001,.999)


def shuffle_paths(path, base, part, rng):
    # Swap complete early/middle logit displacements while keeping the actual
    # late-stage point for each passenger and preserving within-path correlations.
    late = path[part,-1]
    block = (4*base[part]).astype(int)*4 + (4*late).astype(int)
    bank = logit(path[part])
    delta = bank[:,:-1]-bank[:,[-1]]
    for j in np.unique(block):
        indices = np.flatnonzero(block==j)
        delta[indices] = delta[rng.permutation(indices)]
    return expit(np.c_[bank[:,[-1]]+delta,bank[:,-1]])


def run():
    row,y,frame,g,_ = load()
    orig,labels,masks = historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(labels,y)
    OUT.mkdir(exist_ok=True)
    records,history = {},{}
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for field,value in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[field],value)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        draws=np.full((len(y),2,len(EPOCHS)),np.nan)
        bank={}
        for k in np.unique(fold):
            q=np.flatnonzero(fold==k)
            t=np.flatnonzero((fold!=k)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for i,seed in enumerate((1939,7049)):
                draws[q,i]=trajectory(frame,y,t,q,seed+int(k))
            path=draws[q].mean(1)
            for b,c in GroupKFold(3,shuffle=True,random_state=618).split(q,groups=g[q]):
                assert not np.intersect1d(g[q[b]],g[q[c]]).size
                sham=path.copy()
                for part in (b,c):
                    sham[part]=shuffle_paths(path,base[q],part,np.random.default_rng(708))
                models={"late":path[:,[-1]],"all":path,"shuffled":sham,
                        "late_both":draws[q,:,-1]}
                for mode,p in models.items():
                    bank.setdefault("update_"+mode,np.full(len(y),np.nan))[q[c]]=combine(base[q],p,y[q],b,c)
            print(name,k,flush=True)
        path=draws.mean(1)
        for idx,epoch in enumerate(EPOCHS):
            bank[f"epoch_{epoch}"]=path[:,idx]
        results,hits=evaluate(y,base,ref,bank,fold,name=="v131",masks)
        change=((path>=.5)!=(path[:,[-1]]>=.5)).any(1)
        records[name]=dict(comparison=results,changed=int(change.sum()),changed_baseline_wrong=int((change&(ref!=y)).sum()),
            unchanging_baseline_wrong=int((~change&(ref!=y)).sum()))
        if name!="v131":
            for key,value in hits.items():
                history.setdefault(key,[]).append(value)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,trajectories=draws,**bank)
        write(REPORT/"results.json",dict(plans=records))
    stable={}
    for key,values in history.items():
        hit=np.logical_and.reduce([h for h,_ in values])
        stable[key]={k:int((hit&m).sum()) for k,m in masks.items()}
        stable[key]["gain_ci95_pp"]=interval(np.mean([d for _,d in values],0),g)
    write(REPORT/"results.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(1):
        run()
