"""Compare whole bookings by averaging all one-to-one correspondences between companions."""
from math import factorial
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load, strata
from broad_error_vote_tables_n23 import bank
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, evaluate, write
from broad_passenger_transport_n23 import manifest_coordinates
from exaone_validation_data import compact_features
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_booking_assignment"


def companion_cost(dist, temperature):
    """dist[booking, focal companion, donor member]; one donor member is left unmatched."""
    batches, peers, size = dist.shape
    assert peers+1 == size
    if peers == 0:
        return np.zeros((batches,size))
    weights = np.exp(-dist/temperature)
    # dp[mask] sums products for all bijections of the first popcount(mask)
    # companions onto the donor slots in mask. Each donor slot is used once.
    states = 1 << size
    dp = np.zeros((states,batches))
    dp[0] = 1.
    for mask in range(1,states-1):
        i = mask.bit_count()-1
        for j in range(size):
            if mask & (1<<j):
                dp[mask] += dp[mask^(1<<j)] * weights[:,i,j]
    leftovers = (states-1) ^ (1<<np.arange(size))
    return -(temperature/peers)*np.log(dp[leftovers].T/factorial(peers))


def matrices(row, frame, groups):
    # All raw train roster attributes are available, including companions of a
    # development passenger whose own label is outside the development split.
    groups = np.asarray(groups)
    members = pd.Series(np.arange(len(groups))).groupby(groups).agg(list)
    group_size = np.asarray([len(members.at[g]) for g in groups])
    x = manifest_coordinates(frame)
    rng = np.random.default_rng(513)
    a,b = rng.integers(len(x),size=(2,6000))
    temp = np.median(np.sum((x[a]-x[b])**2,axis=1))/5
    out = {}
    for size in np.unique(group_size[row]):
        dev = np.flatnonzero(group_size[row]==size)
        if size==1:
            distance = cdist(x[row[dev]],x[row[dev]],"sqeuclidean").astype(np.float32)
            out[size] = dev, {"own":distance}
            continue
        names = np.unique(groups[row[dev]])
        party = np.array([members.at[g] for g in names])
        slot = np.empty(len(x),int)
        slot[party.ravel()] = np.tile(np.arange(size),len(names))
        mapped = np.searchsorted(names,groups[row[dev]])
        member = slot[row[dev]]
        # Keep the same donor slot while replacing its surrounding party with
        # another entire party of equal size. All assignments are label-blind.
        order = rng.permutation(len(party))
        sham = np.empty(len(party),int)
        sham[order] = np.roll(order,1)
        assert (sham!=np.arange(len(party))).all()
        scores = {k:np.empty((len(dev),len(dev)),np.float32) for k in ("own","means","assignment","sham")}
        donors = x[party]
        donor_means = (donors.sum(1)[:,None,:]-donors)/(size-1)
        for gquery in range(len(names)):
            query = x[party[gquery]]
            costs = cdist(query,donors.reshape(-1,x.shape[1]),"sqeuclidean").reshape(size,len(names),size)
            for i in np.flatnonzero(mapped==gquery):
                focal = member[i]
                remain = np.arange(size)!=focal
                own = costs[focal,mapped,member]
                matched = companion_cost(costs[remain].transpose(1,0,2),temp)
                delta = query[remain].mean(0)-donor_means
                mean_cost = np.sum(delta**2,axis=2)
                scores["own"][i] = own
                scores["assignment"][i] = own+matched[mapped,member]
                scores["sham"][i] = own+matched[sham[mapped],member]
                scores["means"][i] = own+mean_cost[mapped,member]
        out[size] = dev,scores
    return out


def donor_probability(dist,target):
    # Include all ties at the boundary; exclude no donors using outcome values.
    radius = np.partition(dist,min(39,dist.shape[1]-1),axis=1)[:,min(39,dist.shape[1]-1)]
    d = dist-dist.min(1,keepdims=True)
    weight = np.exp(-d/np.maximum(radius[:,None]-dist.min(1,keepdims=True),1e-7))*(dist<=radius[:,None])
    return (weight@target+1)/(weight.sum(1)+2)


def run():
    row,y,local,g,_ = load()
    raw = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ("Name","Transported"))
    groups = raw.PassengerId.str.split("_").str[0].astype(int).to_numpy()
    frame = compact_features(raw)
    np.testing.assert_array_equal(groups[row],g)
    pd.testing.assert_frame_equal(frame.iloc[row].reset_index(drop=True),local.drop(columns="BookingSize"))
    cached = matrices(row,frame,groups)
    labels,deployed,masks = development()
    np.testing.assert_array_equal(labels,y)
    miss = np.logical_and.reduce([p!=y for p in deployed.values()])
    state = strata(local)
    multi = local.BookingSize.to_numpy()>1
    masks = dict(**masks,miss=miss,paid=miss&y&(state==2),zero=miss&~y&(state==0),miss_multi=miss&multi)
    assert [masks[k].sum() for k in ("miss","paid","zero")]==[941,415,281]
    records,history,fallback_counts={},{},{}
    own_x = manifest_coordinates(frame)[row]
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,_,_ = bank(name,row,y)
        scores = {k:np.full(len(y),np.nan) for k in ("own","assignment","means","sham")}
        fallback_counts[name]=0
        for dev,distances in cached.values():
            for f in np.unique(fold):
                q = np.flatnonzero(fold[dev]==f)
                if not len(q):
                    continue
                t = np.flatnonzero((fold[dev]!=f) & ~np.isin(g[dev],g[fold==f]))
                if not len(t):
                    # Some large bookings in the older passenger-random replay
                    # have no same-size reference group outside the entire fold.
                    other = np.flatnonzero((fold!=f)&~np.isin(g,g[fold==f]))
                    assert len(other) and not np.intersect1d(g[dev[q]],g[other]).size
                    d = cdist(own_x[dev[q]],own_x[other],"sqeuclidean")
                    fallback = donor_probability(d,y[other])
                    for score in scores.values():
                        score[dev[q]] = fallback
                    fallback_counts[name] += len(q)
                    continue
                assert not np.intersect1d(g[dev[q]],g[dev[t]]).size
                for key in scores:
                    d = distances.get(key,distances["own"])[np.ix_(q,t)]
                    scores[key][dev[q]] = donor_probability(d,y[dev[t]])
        combined={}
        active = multi & (ref==(base>=.5))
        for key in scores:
            x = np.c_[logit(base.clip(.001,.999)),logit(scores["own"])]
            if key!="own":
                x = np.c_[x,logit(scores[key])-x[:,1]]
            p = base.clip(.001,.999).copy()
            for f in np.unique(fold):
                ids = np.flatnonzero((fold==f)&active)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q = ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q] = LogisticRegression(C=.5,max_iter=500).fit(x[t],y[t]).predict_proba(x[q])[:,1]
            combined["update_"+key] = p
        scores.update(combined)
        records[name],hits = evaluate(y,base,ref,scores,fold,name=="v131",masks)
        for key,(hit,gain) in hits.items():
            if key.startswith("update"):
                assert not gain[~multi].any()
            if name!="v131":
                history.setdefault(key,[]).append(hit)
        out = ART/"nov23_booking_assignment"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out/f"{name}.npz",row=row,y=y,fold=fold,ref=ref,base=base,**scores)
        print(name,[(k,r["fixed"],r["harmed"]) for k,r in records[name].items()],flush=True)
    stable = {k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in history.items()}
    write(REPORT/"results.json",dict(plans=records,stable=stable,fallback=fallback_counts,
          multi_by_scope={s:int((m&multi).sum()) for s,m in masks.items()},multi=int(multi.sum())))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
