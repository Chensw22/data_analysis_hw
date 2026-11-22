"""Compare real companions with passengers' peer profiles from matched, unrelated bookings."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from broad_booking_policy_n23 import EXTRA
from class_analogue_experts_n23 import design
from exaone_validation_data import compact_features
from peer_attribute_surprise_n23 import encode, peer_means
from relational_error_corrector import write

OUT = ART / "nov23_broad_matched_companions"
REPORT = ROOT / "reports/nov23_broad_matched_companions"
BLOCKS = ("route", "sleep_vip", "cabin", "age", "bills")


def data(row):
    raw = pd.concat([pd.read_csv(ROOT / f"data/raw/{name}.csv", usecols=lambda c: c not in ("Transported","Name"))
                     for name in ("train","test")], ignore_index=True)
    group = raw.PassengerId.str.split("_").str[0]
    size = group.map(group.value_counts()).to_numpy()[row]
    frame = compact_features(raw)
    blocks = encode(frame)
    splits = np.cumsum([v.shape[1] for v in blocks])
    fields = np.split(np.arange(splits[-1]), splits[:-1])
    peers = peer_means(np.column_stack(blocks), group.to_numpy())[row]
    return frame.iloc[row].reset_index(drop=True), group.to_numpy()[row], size, peers, fields


def donor_ids(own, group, size, t, query, rng):
    # Conditioning uses each focal passenger's own attributes. Its peer block
    # is deliberately absent from the distance. Donors are training bookings.
    draws = np.full((len(query), 8), -1, int)
    band = np.minimum(size, 4)
    for val in np.unique(band[query]):
        mask = band[query] == val
        ids = query[mask]
        source = t[band[t] == val]
        near = NearestNeighbors(n_neighbors=min(48,len(source)), algorithm="brute").fit(own[source])
        candidates = source[near.kneighbors(own[ids],return_distance=False)]
        for pos, i, neighbors in zip(np.flatnonzero(mask),ids,candidates):
            other = neighbors[group[neighbors] != group[i]]
            # Equal probability per distinct booking among the 12 closest.
            _, unique = np.unique(group[other],return_index=True)
            pool = other[np.sort(unique)[:12]]
            assert len(pool) >= 2
            draws[pos] = rng.choice(pool,size=8,replace=True)
    assert (draws>=0).all() and (group[query,None] != group[draws]).all()
    return draws


def fit(x,y):
    return HistGradientBoostingClassifier(max_iter=170,max_leaf_nodes=15,min_samples_leaf=24,
                  l2_regularization=2,learning_rate=.07,early_stopping=False,random_state=191).fit(x,y)


def logodds(p):
    return logit(np.clip(p,.002,.998))


def fold_scores(frame, group, size, peer, fields, y, t, q, k):
    own, phi = design(frame,t)
    own_x = np.column_stack([phi, np.log(size)])
    rng = np.random.default_rng(919 + int(k))
    tr = donor_ids(own,group,size,t,t,rng)
    te = donor_ids(own,group,size,t,q,rng)
    def assemble(idx,ctx):
        return np.column_stack([own_x[idx],ctx])
    model = fit(assemble(t,peer[t]),y[t])
    def predict(context):
        return model.predict_proba(assemble(q,context))[:,1]
    answer = {"own":fit(own_x[t],y[t]).predict_proba(own_x[q])[:,1],
              "real":predict(peer[q])}
    answer["matched_mean"]=np.mean([predict(peer[j]) for j in te.T],axis=0)
    placebo=fit(assemble(t,peer[tr[:,0]]),y[t])
    answer["retrained_placebo"]=np.mean([placebo.predict_proba(assemble(q,peer[j]))[:,1] for j in te.T],axis=0)
    # Match shrinkage and the number of input columns; isolate the residual
    # of the real peer profile relative to other similar passengers' profiles.
    resid_train=peer[t]-peer[tr].mean(1)
    resid_test=peer[q]-peer[te].mean(1)
    answer["context_residual"]=fit(assemble(t,resid_train),y[t]).predict_proba(assemble(q,resid_test))[:,1]
    for name,cols in zip(BLOCKS,fields):
        draws=[]
        for donor in te.T:
            v=peer[q].copy()
            v[:,cols]=peer[donor][:,cols]
            draws.append(predict(v))
        answer["replace_"+name]=np.mean(draws,axis=0)
    for p in answer.values():
        assert np.isfinite(p).all()
    return answer


def run():
    row,y,base,old,_,_,_=panel()
    ext=np.stack([predictions(row,y,s,EXTRA)[1] for s in SEEDS])
    prior_hit=(np.concatenate([old,ext],axis=2)>=.5)==y[None,:,None]
    frame,group,size,peer,fields=data(row)
    ref=(base>=.5)==y
    repeat=~ref.any(0)
    masks={"all":np.ones(len(y),bool),"any_base_error":~ref.all(0),"repeat":repeat,
           "all46_wrong":repeat & ~prior_hit.any((0,2)),"with_companions":size>1,"solo":size==1}
    OUT.mkdir(exist_ok=True)
    full={}
    for i,seed in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold=z["fold"]
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
        bank={}
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            for key,p in fold_scores(frame,group,size,peer,fields,y,t,q,k).items():
                bank.setdefault(key,np.full(len(y),np.nan))[q]=p
        # Only a real-versus-matched change can alter the reference here.
        delta=logodds(bank["real"])-logodds(bank["matched_mean"])
        delta[size==1]=0
        bank["transfer"]=expit(logodds(base[i])+delta)
        np.testing.assert_array_equal(bank["transfer"][size==1]>=.5,base[i,size==1]>=.5)
        for key,p in bank.items():
            assert np.isfinite(p).all()
            full.setdefault(key,[]).append(p)
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,fold=fold,base=base[i],**bank)
        fix=(bank["transfer"]>=.5)==y
        print(seed,"transfer",int((fix&~ref[i]).sum()),int((~fix&ref[i]).sum()),flush=True)
    def compare(ok,before,mask):
        return dict(rows=int(mask.sum()),accuracy=ok[:,mask].mean(1).tolist(),
                    fixed=(ok&~before)[:,mask].sum(1).tolist(),harmed=(~ok&before)[:,mask].sum(1).tolist(),
                    correct_all_splits=int((ok.all(0)&mask).sum()))
    hits={k:(np.stack(v)>=.5)==y for k,v in full.items()}
    result={}
    for k,ok in hits.items():
        result[k]={label:compare(ok,ref,m) for label,m in masks.items()}
    effect={}
    for k in ["own","matched_mean","retrained_placebo","context_residual",*["replace_"+n for n in BLOCKS]]:
        effect[k]=compare(hits["real"],hits[k],size>1)
    write(REPORT/"result.json",dict(vs_baseline=result,real_vs_alternative_on_accompanied=effect,
          repeat_with_companions=int((repeat&(size>1)).sum()),hard_with_companions=int((masks["all46_wrong"]&(size>1)).sum())))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
