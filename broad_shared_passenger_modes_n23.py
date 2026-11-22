"""Learn transferable feature directions from the cross-covariance of real travel companions."""
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS
from broad_matched_companions_n23 import fit
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from relational_error_corrector import write


def peer_pairs(groups, ids):
    # One unit of mass per multi-person booking, equally divided among directed
    # pairs. Use every member's unlabelled manifest including missing fields.
    buckets=pd.Series(ids).groupby(groups[ids]).apply(np.array)
    src,dst,placebo,weight=[],[],[],[]
    rng=np.random.default_rng(336)
    for count, subsets in buckets.groupby(buckets.map(len)):
        if count<2 or len(subsets)<2:
            continue
        party=np.stack(subsets.to_list())
        party=party[rng.permutation(len(party))]
        shuffled=np.stack([rng.permutation(v) for v in np.roll(party,1,axis=0)])
        left,right=np.where(~np.eye(count,dtype=bool))
        src.append(party[:,left].ravel())
        dst.append(party[:,right].ravel())
        placebo.append(shuffled[:,right].ravel())
        weight.append(np.full(src[-1].size,1/(count*(count-1))))
    a,b,c,w=map(np.concatenate,(src,dst,placebo,weight))
    assert (a!=b).all() and (groups[a]==groups[b]).all() and (groups[a]!=groups[c]).all()
    np.testing.assert_array_equal(np.sort(b),np.sort(c))
    return a,b,c,w/w.sum()


def axes(z,a,b,w):
    delta=z[a]-(w@z[a])
    other=z[b]-(w@z[b])
    cross=delta.T@(w[:,None]*other)
    gram=delta.T@(w[:,None]*delta)+.15*np.eye(z.shape[1])
    values,directions=eigh((cross+cross.T)/2,gram)
    cols=np.r_[np.arange(6),np.arange(len(values)-10,len(values))]
    return directions[:,cols]


def agreement(z,axis,pairs):
    a,b,c,w=pairs
    u=z@axis
    first=u[a]-w@u[a]
    def corr(ids):
        second=u[ids]-w@u[ids]
        return (w@(first*second))/np.sqrt((w@(first**2))*(w@(second**2))).clip(1e-8)
    return dict(actual=corr(b).tolist(),shuffled=corr(c).tolist())


def run():
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    size=pd.Series(groups).groupby(groups).transform("size").to_numpy()
    frame=compact_features(raw)
    result,eval_corr={},{}
    output=ART/"nov23_broad_shared_passenger_modes"
    output.mkdir(exist_ok=True)
    bank={}
    bases=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            row,y,base,fold=[ref[n] for n in ("row","y","base","fold")]
        bases.append((base>=.5)==y)
        preds={n:np.full(len(y),np.nan) for n in ("own","real_modes","fake_modes","real_only","transfer")}
        correlations=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(groups[row[t]],groups[row[q]]).size
            # The current evaluation bookings and all their companions are removed
            # from the self-supervised fit. Remaining public roster fields carry no labels.
            held=np.isin(groups,groups[row[q]])
            unsup=np.flatnonzero(~held)
            pairs=peer_pairs(groups,unsup)
            valid_pairs=peer_pairs(groups,np.flatnonzero(held))
            own,phi=design(frame,row[t])
            center=own[unsup].mean(0)
            scale=own[unsup].std(0).clip(.25)
            standardized=((own-center)/scale).clip(-5,5)
            rng=np.random.default_rng(589)
            omega=rng.normal(size=(own.shape[1],96))/np.sqrt(own.shape[1])
            z=np.column_stack([standardized,np.sqrt(2)*np.cos(standardized@omega+rng.uniform(0,2*np.pi,96))])
            a,b,c,w=pairs
            real=axes(z,a,b,w)
            fake=axes(z,a,c,w)
            correlations.append({"learned_real":agreement(z,real,valid_pairs),
                                 "learned_fake":agreement(z,fake,valid_pairs)})
            own_x=np.column_stack([phi[row],np.log(size[row])])
            embeddings=z[row]@real
            inputs={"own":own_x,"real_modes":np.column_stack([own_x,embeddings]),
                    "fake_modes":np.column_stack([own_x,z[row]@fake]),"real_only":embeddings}
            for n,x in inputs.items():
                preds[n][q]=fit(x[t],y[t]).predict_proba(x[q])[:,1]
            delta=logit(preds["real_modes"][q].clip(.002,.998))-logit(preds["own"][q].clip(.002,.998))
            preds["transfer"][q]=expit(logit(base[q].clip(.002,.998))+delta)
        eval_corr[seed]=correlations
        assert all(np.isfinite(v).all() for v in preds.values())
        np.savez_compressed(output/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**preds)
        for name,v in preds.items():
            bank.setdefault(name,[]).append((v>=.5)==y)
        print(seed,{n:round(float(((p>=.5)==y).mean()),4) for n,p in preds.items()},flush=True)
    bases=np.stack(bases)
    repeated=~bases.any(0)
    for name,arr in bank.items():
        correct=np.stack(arr)
        result[name]={}
        for label,mask in {"all":np.ones(len(y),bool),"solo":size[row]==1,"accompanied":size[row]>1,
                           "ever_wrong":~bases.all(0),"repeat":repeated,"repeat_solo":repeated&(size[row]==1)}.items():
            result[name][label]=dict(rows=int(mask.sum()),accuracy=correct[:,mask].mean(1).tolist(),
                fixed=(correct[:,mask]&~bases[:,mask]).sum(1).tolist(),harmed=(~correct[:,mask]&bases[:,mask]).sum(1).tolist(),
                correct_all=int((correct.all(0)&mask).sum()))
    write(ROOT/"reports/nov23_broad_shared_passenger_modes/result.json",dict(scores=result,held_out_pair_correlations=eval_corr))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
