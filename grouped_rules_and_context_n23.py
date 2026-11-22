"""Combine a sparse-rule classifier with a complementary in-context prior at fixed weights."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ROOT,ART,SEEDS
from grouped_five_way_coverage_n23 import verify
from grouped_sparse_rules_n23 import remaining
from relational_error_corrector import write


def run():
    data=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    rows,y=data.row.to_numpy(),data.target.to_numpy(bool)
    hard=remaining(rows,y)
    groups=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.iloc[rows].str.split("_").str[0]
    _,group=np.unique(groups,return_inverse=True)
    out=ART/"nov23_rule_context_pool"
    out.mkdir(exist_ok=True)
    summary={}
    correct=[]
    base_correct=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_sparse_rules/{seed}.npz") as z:
            base,fold,rules=z["base"],z["fold"],z["rules"]
            verify(z,rows,y,fold)
        with np.load(ART/f"nov23_tabicl_context/{seed}.npz") as z:
            verify(z,rows,y,fold)
            np.testing.assert_allclose(base,z["base"])
            global_icl=z["global_context"]
            local_icl=z["local"]
        # One quarter of total weight goes to each new family; halves of a local/global
        # ICL pair share that family's existing quarter. No fit or score-based weight search.
        variants={"rule_control":.75*base+.25*rules,"global":.5*base+.25*(rules+global_icl),
                  "global_local":.5*base+.25*rules+.125*(local_icl+global_icl)}
        old=(base>=.5)==y
        base_correct.append(old)
        panel=[]
        metrics={}
        for name,p in variants.items():
            ok=(p>=.5)==y
            metrics[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&~old).sum()),harmed=int((~ok&old).sum()),
                               hard356_correct=int(ok[hard].sum()))
            panel.append(ok)
        correct.append(panel)
        summary[str(seed)]=metrics
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**variants)
    hit=np.asarray(correct)
    base=np.asarray(base_correct)
    stable={}
    for j,name in enumerate(variants):
        delta=(hit[:,j].astype(float)-base).mean(axis=0)
        gain=np.bincount(group,weights=delta)
        sizes=np.bincount(group)
        rng=np.random.default_rng(1511)
        draws=[]
        for _ in range(80):
            counts=rng.multinomial(len(sizes),np.full(len(sizes),1./len(sizes)),size=50)
            draws.extend(100*(counts@gain)/(counts@sizes))
        stable[name]=dict(mean_gain_pp=float(100*delta.mean()),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                          stable_fixes=int((hit[:,j].all(0)&~base.any(0)).sum()),
                          stable_harms=int((~hit[:,j].any(0)&base.all(0)).sum()))
    write(ROOT/"reports/nov23_rule_context_pool/result.json",dict(results=summary,stability=stable))
    print(summary,stable,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
