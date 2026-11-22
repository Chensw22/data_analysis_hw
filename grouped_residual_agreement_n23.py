"""Estimate trust in residual corrections from sensitivity to training bookings."""
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features
from group_crossfit_residual_discovery_n23 import teachers
from grouped_field_factors_n23 import encode
from grouped_five_way_coverage_n23 import verify
from grouped_teacher_field_offsets_n23 import fit_correction
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def decisions(base,fold,group,y,members):
    # Both referees consider exactly the same complaints; the control sees the
    # same aggregate teacher outcome and base margins but no stability fields.
    avg=members.mean(1)
    forward=base>=.5
    eligible=(avg>=.5)!=forward
    wrong=y!=forward
    signed=np.where(forward[:,None],1-members,members)
    mean=signed.mean(1)
    simple=np.column_stack([np.abs(logit(base.clip(.001,.999))),mean,forward])
    votes=(signed>.5).mean(1)
    deviation=signed.std(1)
    full=np.column_stack([simple,votes,deviation,votes*(mean-.5),deviation*(mean-.5)])
    output={key:base.copy() for key in ["mean_referee","agreement_referee"]}
    for k in np.unique(fold):
        outer=np.flatnonzero(fold==k)
        for a,b in GroupKFold(3,shuffle=True,random_state=455).split(outer,groups=group[outer]):
            t,q=outer[a],outer[b]
            assert not np.intersect1d(group[t],group[q]).size
            t,q=t[eligible[t]],q[eligible[q]]
            for key,matrix in [("mean_referee",simple),("agreement_referee",full)]:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=500))
                model.fit(matrix[t],wrong[t])
                switch=model.predict_proba(matrix[q])[:,1]
                output[key][q]=np.where(forward[q],1-switch,switch)
    return {"member_mean":avg,"fixed_blend":.75*base+.25*avg,**output}


def run():
    torch.set_num_threads(2)
    panel=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=panel.row.to_numpy(copy=True),panel.target.to_numpy(bool)
    truth,b,c=read_panel()
    np.testing.assert_array_equal(truth,y)
    old=~b.any(0)
    hard=old&~c.any(axis=(0,1))
    original=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_teacher_field_offsets/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],dev)
            np.testing.assert_array_equal(z["y"],y)
            original.append((z["offset3"]>=.5)==y)
    recovered=hard&np.any(original,axis=0)
    assert old.sum()==1070 and hard.sum()==466 and recovered.sum()==149
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Transported","Name"]) for s in ["train","test"]],ignore_index=True)
    party=raw.PassengerId.str.split("_").str[0]
    group=party.iloc[dev].to_numpy()
    frame=compact_features(raw.iloc[dev]).reset_index(drop=True)
    frame["BookingSize"]=party.map(party.value_counts()).to_numpy()[dev]
    x=encode(raw)[dev]
    target=torch.from_numpy(y.astype(np.float32))
    out=ART/"nov23_residual_agreement"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            base,fold=ref["base"],ref["fold"]
            verify(ref,dev,y,fold)
        members=np.full((len(y),3),np.nan)
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            bookings=np.unique(group[t])
            for i in range(members.shape[1]):
                rng=np.random.default_rng(seed+2903*i+int(k))
                selected=rng.choice(bookings,int(.8*len(bookings)),replace=False)
                subset=t[np.isin(group[t],selected)]
                train,query=teachers(frame,y,group,subset,q,seed+int(k)+i*97)
                ts,qs=[logit(a.clip(.001,.999)) for a in [train,query]]
                members[q,i]=fit_correction(x,target,subset,q,3,ts,qs,0)
        assert np.isfinite(members).all()
        preds=decisions(base,fold,group,y,members)
        metrics={}
        for name,p in preds.items():
            good=(p>=.5)==y
            bad=(base>=.5)!=y
            metrics[name]=dict(accuracy=float(good.mean()),fixed=int((good&bad).sum()),harmed=int((~good&~bad).sum()),
                               recovered1070=int((good&old).sum()),recovered466=int((good&hard).sum()),
                               recovered149=int((good&recovered).sum()))
        results[str(seed)]=metrics
        write(ROOT/"reports/nov23_residual_agreement/result.json",results)
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,members=members,**preds)
        print(seed,metrics,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
