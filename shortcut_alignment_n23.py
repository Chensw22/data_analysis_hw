"""Learn label-conditioned representations across easy and misleading feature views."""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from torch import nn

from class_analogue_experts_n23 import design
from conditional_view_density_n23 import fit_score
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_error_accounting_n23 import ART, BILLS, ROOT, SEEDS
from relational_error_corrector import write


def difficulty(frame, labels, groups, train, seed):
    bills = np.log1p(frame[BILLS].to_numpy())
    registry = pd.get_dummies(frame[CATEGORICAL], dtype=float)
    registry["age"] = frame.Age
    registry["berth"] = frame.CabinNumber
    guesses = np.full((len(train),2), np.nan)
    for a,b in GroupKFold(3,shuffle=True,random_state=seed).split(train,groups=groups[train]):
        learn,held = train[a], train[b]
        assert not np.intersect1d(groups[learn], groups[held]).size
        for j,x in enumerate((bills,registry.to_numpy())):
            guesses[b,j] = fit_score(x[learn],labels[learn],x[held],None)
    assert np.isfinite(guesses).all()
    return (guesses>=0) != labels[train,None]


def train_network(x, labels, misses, query, mode):
    torch.manual_seed(149)
    encoder = nn.Sequential(nn.Linear(x.shape[1],80),nn.LayerNorm(80),nn.SiLU(),
                            nn.Linear(80,24),nn.LayerNorm(24),nn.SiLU())
    readout = nn.Linear(24,1)
    opt = torch.optim.AdamW([*encoder.parameters(),*readout.parameters()],lr=.0018,weight_decay=.02)
    x, query = (torch.from_numpy(a.astype(np.float32)) for a in (x,query))
    rng = np.random.default_rng(103)
    flags = misses.copy()
    if mode=="shuffled":
        shuffler=np.random.default_rng(421)
        for cls in (False,True):
            idx=np.flatnonzero(labels==cls)
            flags[idx]=flags[shuffler.permutation(idx)]
    flags=torch.from_numpy(flags)
    y=torch.from_numpy(labels.copy())
    snapshots=[]
    for epoch in range(28):
        for idx in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/320))):
            z=encoder(x[idx])
            loss=nn.functional.binary_cross_entropy_with_logits(readout(z).ravel(),y[idx].float())
            if mode!="plain":
                # For each actual label, align the first two moments of features
                # across passengers on whom each shortcut succeeds or fails.
                differences=[]
                features=nn.functional.normalize(z,dim=1)
                for cls in (False,True):
                    for split in range(flags.shape[1]):
                        m=y[idx]==cls
                        f=flags[idx,split]
                        a,b=features[m & f],features[m & ~f]
                        if min(len(a),len(b))>=4:
                            differences.append((a.mean(0)-b.mean(0)).square().sum()
                                +(a.T@a/len(a)-b.T@b/len(b)).square().sum())
                loss=loss+.45*torch.stack(differences).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in (19,23,27):
            with torch.no_grad():
                snapshots.append(torch.sigmoid(readout(encoder(query))).ravel().numpy())
    return np.mean(snapshots,axis=0)


def run():
    torch.set_num_threads(2)
    inventory=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    rows,y=inventory.row.to_numpy(),inventory.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in {"Transported","Name"})
                   for part in ("train","test")],ignore_index=True)
    bookings=raw.PassengerId.str.split("_").str[0]
    size=bookings.map(bookings.value_counts()).to_numpy()[rows]
    group=bookings.to_numpy()[rows]
    frame=compact_features(raw.iloc[rows]).reset_index(drop=True)
    artifact=ART/"nov23_shortcut_alignment"
    artifact.mkdir(exist_ok=True)
    with np.load(ART/"nov23_archive_recovery/masks.npz") as masks:
        np.testing.assert_array_equal(rows,masks["row"])
        hard=masks["originally_unresolved"]
        deepest=masks["still_unresolved"]
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            base,fold=ref["base"],ref["fold"]
            verify(ref,rows,y,fold)
        scores={mode:np.full(len(y),np.nan) for mode in ("plain","aligned","shuffled")}
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(group[t],group[q]).size
            _,phi=design(frame,t)
            phi=np.column_stack((phi,np.log1p(size)))
            difficulty_flags=difficulty(frame,y,group,t,seed+int(f))
            for mode,out in scores.items():
                out[q]=train_network(phi[t],y[t],difficulty_flags,phi[q],mode)
            print(seed,"fold",f,flush=True)
        summary={}
        for name,p in scores.items():
            assert np.isfinite(p).all()
            for kind,s in [(name,p),(name+"_quarter",.75*base+.25*p)]:
                correct=(s>=.5)==y
                baseline=(base>=.5)==y
                summary[kind]=dict(accuracy=float(correct.mean()),fixed=int((correct & ~baseline).sum()),
                                   harmed=int((~correct & baseline).sum()),correct_on_365=int(correct[hard].sum()),
                                   correct_on_252=int(correct[deepest].sum()),
                                   positive_on_365=int((correct & hard & y).sum()))
        result[seed]=summary
        np.savez_compressed(artifact/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,**scores)
        write(ROOT/"reports/nov23_shortcut_alignment/result.json",dict(results=result))
        print(seed,summary["aligned"],summary["aligned_quarter"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
