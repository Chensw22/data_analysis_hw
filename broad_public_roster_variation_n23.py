"""Check paired historical score shifts against variation in the size-matched booking roster."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import load
from broad_partition_evidence_n23 import ROOT, write
from broad_twin_surfaces_n23 import REPORT
from released_correction_audit_n23 import development, release, REPORT as RECEIPTS


def groups_of(ids):
    return ids.str.split("_").str[0].astype(int).to_numpy()


def count_groups(ids):
    _,sizes=np.unique(ids,return_counts=True)
    return np.bincount(sizes,minlength=9)


def simulate(tables, frequencies, reps, rng):
    estimate=np.zeros(tables[1].shape[1])
    noise=np.zeros((reps,len(estimate)))
    roster=sum(n*k for k,n in enumerate(frequencies))
    for size,data in tables.items():
        n=frequencies[size]
        if not n:
            continue
        estimate+=n*data.mean(0)/roster
        # Sample both a hypothetical leaderboard roster and the development
        # bookings that were used to estimate each size stratum's mean.
        kinds,counts=np.unique(data,axis=0,return_counts=True)
        prob=counts/counts.sum()
        test=rng.multinomial(n,prob,size=reps)@kinds
        dev=rng.multinomial(len(data),prob,size=reps)@kinds
        noise+=(test-(n/len(data))*dev)/roster
    return estimate,noise,int(roster)


def main():
    row,y,frame,g,_=load()
    targets,pred,_=development()
    np.testing.assert_array_equal(y,targets)
    ids=pd.read_csv(ROOT/"data/raw/test.csv",usecols=["PassengerId"]).PassengerId
    official=json.loads((RECEIPTS/"primary_receipts.json").read_text())["scores"]
    releases={n:release(n,ids,official) for n in pred}
    versions=list(range(114,137))
    baseline=[108 if v<129 else 128 for v in versions]
    gains=np.column_stack([(pred[v]==y).astype(int)-(pred[b]==y).astype(int) for v,b in zip(versions,baseline)])
    public=np.array([releases[v][0]-releases[b][0] for v,b in zip(versions,baseline)])
    flips=np.array([(releases[v][1]!=releases[b][1]).sum() for v,b in zip(versions,baseline)])
    group,inv,n=np.unique(g,return_inverse=True,return_counts=True)
    sizes=frame.BookingSize.to_numpy(dtype=int)
    full=n[inv]==sizes
    testgroups=groups_of(ids)
    histogram=count_groups(testgroups)
    assert sum(k*v for k,v in enumerate(histogram))==len(ids)==4277
    assert not np.intersect1d(g,testgroups).size
    totals=np.zeros((len(group),len(versions)),dtype=int)
    np.add.at(totals,inv,gains)
    tables={}
    changed_groups=np.zeros(len(versions),int)
    for size in np.flatnonzero(histogram):
        scope=full&(sizes==size)
        take=np.unique(inv[scope])
        assert (n[take]==size).all()
        tables[size]=totals[take]
        any_flip=np.zeros((len(group),len(versions)),int)
        np.add.at(any_flip,inv[scope],np.abs(gains[scope]))
        changed_groups+=(any_flip[take]>0).sum(0)
    reports={}
    for frac in (1.,.5,.25):
        counts=np.rint(histogram*frac).astype(int)
        mid,noise,nrows=simulate(tables,counts,3000,np.random.default_rng(719))
        sd=noise.std(0,ddof=1)
        valid=sd>1.e-12
        cutoff=np.quantile(np.max(np.abs(noise[:,valid]/sd[valid]),axis=1),.95)
        lo,hi=mid+np.quantile(noise,[.025,.975],axis=0)
        margin=.000011  # Score differences inherit two displayed scores' precision.
        outside=(public<lo-margin)|(public>hi+margin)
        family=np.abs(public-mid)>cutoff*sd+margin
        reports[str(frac)]=dict(rows=nrows,marginal_outside=int(outside[valid].sum()),
                simultaneous_outside=int(family[valid].sum()),estimated=int(valid.sum()),
                contrasts=[dict(version=v,anchor=b,local_all_pp=100*float(gains[:,j].mean()),
                    roster_adjusted_pp=100*float(mid[j]), public_pp=100*float(public[j]),
                    changed_complete_dev_groups=int(changed_groups[j]),changed_test_rows=int(flips[j]),
                    conditional_band_pp=[100*float(lo[j]),100*float(hi[j])],
                    simultaneous_outside=bool(family[j])) for j,(v,b) in enumerate(zip(versions,baseline))])
        print(frac,nrows,int(outside[valid].sum()),int(family[valid].sum()),np.array(versions)[family],flush=True)
    write(REPORT/"public_roster.json",dict(versions=len(versions),complete_people=int(full.sum()),
            fully_labeled_groups_by_size={str(k):len(v) for k,v in tables.items()},
            submission_groups_by_size=histogram.tolist(),scenarios=reports))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
