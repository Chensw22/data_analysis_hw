"""Test whether the order of bookings on other decks and sides predicts an individual's outcome."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_refit_dissent_n23 import features
from broad_training_membership_n23 import model
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes

REPORT = ROOT / "reports/nov23_manifest_exposure"


def stream_context():
    roster = pd.concat([pd.read_csv(ROOT/f"data/raw/{split}.csv", usecols=["PassengerId","Cabin"])
                        for split in ("train","test")],ignore_index=True)
    ids = roster.PassengerId.str.split("_",expand=True).astype(int)
    group = ids[0].to_numpy()
    positions = np.unique(group)
    cabins = roster.Cabin.str.split("/",expand=True)
    stream = (cabins[0]+"/"+cabins[2]).fillna("?").to_numpy()
    channels = sorted(set(stream)-{"?"})
    exposure, lookup = [], {}
    for i, channel in enumerate(channels):
        # A booking can contain several passengers in the same stream and
        # contributes just one event. Strict inequalities exclude the focal booking.
        events = np.unique(group[stream==channel])
        low = np.searchsorted(events,positions,side="left")
        high = np.searchsorted(events,positions,side="right")
        gaps = []
        for k in (1, 4):
            prev, after = low-k, high+k-1
            left = np.where(prev>=0,positions-events[prev.clip(0,len(events)-1)],np.nan)
            right = np.where(after<len(events),events[after.clip(0,len(events)-1)]-positions,np.nan)
            for v in (left,right):
                assert np.all(v[np.isfinite(v)]>0)
                gaps.append(np.log1p(v))
        exposure.append(np.column_stack(gaps))
        lookup[channel] = i
    bank = np.stack(exposure,axis=1)
    idx = np.searchsorted(positions,group)
    own = np.array([lookup.get(c,-1) for c in stream])
    def assemble(c):
        others = c[idx].copy()
        mine = np.full((len(group),c.shape[-1]),np.nan)
        rows = np.flatnonzero(own>=0)
        mine[rows] = others[rows,own[rows]]
        others[rows,own[rows]] = np.nan
        return mine, others.reshape(len(group),-1)
    # Block permutation jointly shuffles information from all channels, while
    # preserving their marginal values and correlations at nearby roster positions.
    perm = np.arange(len(positions));rng = np.random.default_rng(827)
    for start in range(0,len(perm),128):
        ix = perm[start:start+128].copy()
        perm[start:start+128] = rng.permutation(ix)
    own_fields, other_fields = assemble(bank)
    _, sham = assemble(bank[perm])
    prior = np.array([len(np.unique(group[stream==c])) for c in channels],float)
    prior /= prior.sum()
    diagnostic = {}
    known = np.flatnonzero(own>=0)
    for key,values in (("actual",bank),("shuffled",bank[perm])):
        span = np.expm1(values[:,:,2])+np.expm1(values[:,:,3])
        intensity = np.divide(7.,span,out=np.full_like(span,np.nan),where=span>0)
        intensity = np.where(np.isfinite(intensity),intensity,prior)
        prob = intensity/intensity.sum(1,keepdims=True)
        diagnostic[key] = dict(mean_log_loss=float(-np.log(prob[idx[known],own[known]]).mean()),
                               top_one=float((prob[idx[known]].argmax(1)==own[known]).mean()))
    diagnostic["global"] = dict(mean_log_loss=float(-np.log(prior[own[known]]).mean()),
                                top_one=float((own[known]==prior.argmax()).mean()))
    return group,ids[1].to_numpy(),own_fields,other_fields,sham,len(channels),diagnostic


def main():
    row,y,frame,groups,_ = load()
    group,member,own,other,sham,nstreams,diagnostic = stream_context()
    np.testing.assert_array_equal(groups,group[row])
    x = np.c_[features(frame)[0],group[row],member[row]]
    baseline = np.c_[x,own[row]]
    designs = {"basic":x,"own":baseline,"other":np.c_[baseline,other[row]],"shuffled":np.c_[baseline,sham[row]]}
    masks = scopes(y,frame)
    out = ART/"nov23_manifest_exposure"
    out.mkdir(exist_ok=True)
    histories,refs = {},[]
    for s in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{s}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold=z["base"],z["fold"]
        scores = {}
        for key,design in designs.items():
            p=np.full(len(y),np.nan)
            for f in np.unique(fold):
                t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
                assert not np.intersect1d(groups[t],groups[q]).size
                p[q]=model(160).fit(design[t],y[t]).predict_proba(design[q])[:,1]
            scores[key]=p
            z=np.c_[logit(base.clip(.001,.999)),logit(p.clip(.001,.999))]
            combined=np.full(len(y),np.nan)
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=groups[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(groups[t],groups[q]).size
                    combined[q]=LogisticRegression(C=.5,max_iter=400).fit(z[t],y[t]).predict_proba(z[q])[:,1]
            scores["combined_"+key]=combined
        ref=(base>=.5)==y
        refs.append(ref)
        for k,p in scores.items():
            assert np.isfinite(p).all()
            histories.setdefault(k,[]).append((p>=.5)==y)
        np.savez_compressed(out/f"{s}.npz",row=row,y=y,fold=fold,base=base,**scores)
        print(s,{k:round(float(h[-1].mean()),4) for k,h in histories.items()},flush=True)
    stats={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in histories.items()}
    paired = {}
    for method,control in (("other","shuffled"),("combined_other","combined_shuffled"),("other","basic")):
        diff = np.mean(histories[method],axis=0)-np.mean(histories[control],axis=0)
        paired[method+"_vs_"+control] = dict(gain_pp=float(100*diff.mean()),interval_pp=interval(diff,groups))
    write(REPORT/"results.json",dict(streams=nstreams,results=stats,channel_diagnostic=diagnostic,paired=paired))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
