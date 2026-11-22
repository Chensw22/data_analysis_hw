"""Train low-rank field interactions as corrections to inner group-held-out teacher logits."""
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from torch import nn
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from grouped_field_factors_n23 import FieldFactors,encode
from group_crossfit_residual_discovery_n23 import teachers
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


def fit_correction(x,y,t,q,order,train_offset,query_offset,strength):
    torch.manual_seed(733)
    model=FieldFactors(*x.shape[1:],order)
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.03)
    train_offset=torch.tensor(train_offset,dtype=torch.float32)
    query_offset=torch.tensor(query_offset,dtype=torch.float32)
    rng=np.random.default_rng(733)
    estimates=[]
    for epoch in range(36):
        for indices in np.array_split(rng.permutation(len(t)),int(np.ceil(len(t)/384))):
            d=model(x[t[indices]])
            z=train_offset[indices]+d
            loss=nn.functional.binary_cross_entropy_with_logits(z,y[t[indices]])+strength*d.square().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2)
            opt.step()
        if epoch in [23,29,35]:
            with torch.no_grad():
                d=torch.cat([model(x[part]) for part in np.array_split(q,8)])
                estimates.append(torch.sigmoid(query_offset+d[:,None]).mean(1).numpy())
    return np.mean(estimates,axis=0)


def run():
    torch.set_num_threads(2)
    rows=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    dev,y=rows.row.to_numpy(copy=True),rows.target.to_numpy(bool)
    test_y,old,others=read_panel()
    np.testing.assert_array_equal(test_y,y)
    old_wrong=~old.any(0)
    hard=old_wrong&~others.any(axis=(0,1))
    assert old_wrong.sum()==1070 and hard.sum()==466
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ["Transported","Name"])
                   for s in ["train","test"]],ignore_index=True)
    party=raw.PassengerId.str.split("_").str[0]
    group=party.iloc[dev].to_numpy()
    features=compact_features(raw.iloc[dev]).reset_index(drop=True)
    features["BookingSize"]=party.map(party.value_counts()).to_numpy()[dev]
    x,target=encode(raw)[dev],torch.from_numpy(y.astype(np.float32))
    out=ART/"nov23_teacher_field_offsets"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_field_factors/{seed}.npz") as z:
            base,fold,direct=z["base"],z["fold"],z["order3"]
            verify(z,dev,y,fold)
        variants=[("offset2",2,0),("offset3",3,0),("penalized_offset3",3,.125)]
        pred={name:np.full(len(y),np.nan) for name in [v[0] for v in variants]+["teacher"]}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            train,query=teachers(features,y,group,t,q,seed+int(k))
            pred["teacher"][q]=query.mean(1)
            train_logit,query_logit=[logit(v.clip(.001,.999)) for v in [train,query]]
            for name,order,strength in variants:
                pred[name][q]=fit_correction(x,target,t,q,order,train_logit,query_logit,strength)
        pred["direct3"]=direct
        pred["teacher_direct3_vote"]=.75*pred["teacher"]+.25*direct
        with np.load(ART/f"nov23_residual_discovery/{seed}.npz") as check:
            verify(check,dev,y,fold)
            np.testing.assert_allclose(check["teacher"],pred["teacher"],rtol=1e-6,atol=1e-6)
        summary={}
        for key,p in pred.items():
            assert np.isfinite(p).all()
            for name,s in [(key,p),(key+"_blend",.75*base+.25*p)]:
                ok=(s>=.5)==y
                bad=(base>=.5)!=y
                teacher_bad=(pred["teacher"]>=.5)!=y
                summary[name]=dict(accuracy=float(ok.mean()),fixed=int((ok&bad).sum()),harmed=int((~ok&~bad).sum()),
                    teacher_fixed=int((ok&teacher_bad).sum()),teacher_harmed=int((~ok&~teacher_bad).sum()),
                    recovered1070=int(ok[old_wrong].sum()),recovered466=int(ok[hard].sum()))
        results[str(seed)]=summary
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**pred)
        write(ROOT/"reports/nov23_teacher_field_offsets/result.json",results)
        print(seed,[(k,round(v["accuracy"]*100,3),v["recovered466"]) for k,v in summary.items()],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
