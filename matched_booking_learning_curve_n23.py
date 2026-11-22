"""On identical unseen bookings compare 4,636 training passengers to seven eighths of development."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
ART = ROOT/"artifacts"
OUT = ART/"nov23_matched_learning_curve"
SEED = 20261123


def portions(groups):
    for k,(t,q) in enumerate(GroupKFold(8,shuffle=True,random_state=SEED).split(groups,groups=groups)):
        unique, counts=np.unique(groups[t],return_counts=True)
        order=np.random.default_rng(SEED+k).permutation(len(unique))
        end=int(np.argmin(abs(np.cumsum(counts[order])-4636)))+1
        small=t[np.isin(groups[t],unique[order[:end]])]
        assert not np.intersect1d(groups[t],groups[q]).size
        yield k, small,t,q


def run(family):
    inventory = pd.read_csv(ART/"nov23_global_error_inventory.csv")
    rows,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Transported","Name"]).iloc[rows].reset_index(drop=True)
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    OUT.mkdir(exist_ok=True)
    if family=="v44":
        import audit_top5_references as audit
        framework,_,fit,_,version=audit.initialize_reference("v44", OUT)
        matrices,dev,labels,parties,_=audit.load_audit_data(framework)
        np.testing.assert_array_equal(rows,dev)
        np.testing.assert_array_equal(group,parties)
        np.testing.assert_array_equal(y,labels)
        def predict(s,q,k):
            p,_=fit(version,matrices,rows[s],labels[s],rows[q],SEED+k)
            return p
    else:
        import hashlib
        import torch
        from tabicl_routed_context_n23 import infer,CAT,DIGEST,WEIGHT
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
        torch.set_num_threads(2)
        frame=raw.drop(columns=["Cabin","PassengerId"]).copy()
        frame[["Deck","CabinNumber","Side"]]=raw.Cabin.str.split("/",expand=True)
        frame.CabinNumber=pd.to_numeric(frame.CabinNumber)
        frame[CAT]=frame[CAT].fillna("__MISSING__").astype(str)
        def predict(s,q,k):
            return infer(frame,y,s,q,SEED+k)
    small,large=np.full(len(y),np.nan),np.full(len(y),np.nan)
    fold=np.full(len(y),-1)
    sizes=[]
    for k,s,t,q in portions(group):
        small[q]=predict(s,q,k)
        large[q]=predict(t,q,k)
        fold[q]=k
        sizes.append([len(s),len(t),len(q)])
        print(f"{family} {k+1}/8: {len(s)} vs {len(t)} support, {len(q)} validation",flush=True)
    assert (fold>=0).all() and np.isfinite([small,large]).all()
    np.savez_compressed(OUT/f"{family}.npz",row=rows,y=y,fold=fold,small=small,large=large,sizes=sizes)


def summarize():
    from population_rescue_map_n23 import read_panel
    from relational_error_corrector import write

    y,existing,candidates=read_panel()
    hard=~existing.any(0)&~candidates.any(axis=(0,1))
    base={}
    for family in ["v44","tabicl"]:
        with np.load(OUT/f"{family}.npz") as z:
            if family=="v44":
                rows,fold,sizes=z["row"],z["fold"],z["sizes"]
            np.testing.assert_array_equal(z["row"],pd.read_csv(ART/"nov23_global_error_inventory.csv").row)
            for key,ref in [("row",rows),("y",y),("fold",fold),("sizes",sizes)]:
                np.testing.assert_array_equal(z[key],ref)
            base[family]=np.array([z["small"],z["large"]])
    base["equal_ensemble"]=(base["v44"]+base["tabicl"])/2
    groups = pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.iloc[rows].str.split("_").str[0]
    _, group_ids = np.unique(groups,return_inverse=True)
    party_sizes = np.bincount(group_ids)
    result={}
    for name,probs in base.items():
        small,large=(probs>=.5)==y
        changes = np.bincount(group_ids,weights=large.astype(float)-small)
        rng = np.random.default_rng(1057)
        draws = []
        for _ in range(80):
            freq = rng.multinomial(len(party_sizes),np.full(len(party_sizes),1/len(party_sizes)),size=50)
            draws.extend(100*(freq@changes)/(freq@party_sizes))
        result[name]=dict(gain_pp=float(100*(large.mean()-small.mean())),
                          paired_booking_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),accuracy_small=float(small.mean()),accuracy_large=float(large.mean()),
                          fixed=int((large&~small).sum()),harmed=int((~large&small).sum()),
                          unresolved466_small=int(small[hard].sum()),unresolved466_large=int(large[hard].sum()),
                          persistent1070_small=int(small[~existing.any(0)].sum()),persistent1070_large=int(large[~existing.any(0)].sum()),
                          fold_net=[int(large[fold==k].sum()-small[fold==k].sum()) for k in range(8)])
    write(ROOT/"reports/nov23_matched_learning_curve/result.json",dict(seed=SEED,results=result,sizes=sizes.tolist(),
          sizes_columns=["small_training_rows","large_training_rows","same_validation_rows"]))
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["v44","tabicl","summarize"])
    phase=parser.parse_args().phase
    if phase=="summarize":
        summarize()
    else:
        run(phase)
