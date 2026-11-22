"""Calibrate the sensitivity of the matched-booking learning curve to a hypothetical additional signal."""
import numpy as np
from threadpoolctl import threadpool_limits
from broad_error_calibration_size_n23 import (REPORT,SEEDS,GroupKFold,cache,fit,load,problem,write)


def run():
    row,y,_,group,_=load()
    output={}
    for name in [*map(str,SEEDS),"v131"]:
        views,anchor,fold,ref=cache(name,row,y)
        x,off,miss=problem(anchor,views,y)
        unlocked=ref==(anchor>=0)
        report={}
        for reliability in (.5,.7,.75):
            weighted={}
            # One observation per passenger available when training the simulated detector;
            # integrate over the two possible observations when calculating its test accuracy.
            noise=np.random.default_rng(648).random(len(y))
            bit=np.where(noise<reliability,miss,~miss).astype(float)
            train_problem=(np.c_[x,bit*2-1],off,miss)
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for j,(a,b) in enumerate(GroupKFold(8,shuffle=True,random_state=631).split(ids,groups=group[ids])):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(group[t],group[q]).size
                    unique,counts=np.unique(group[t],return_counts=True)
                    for draw in range(3):
                        order=np.random.default_rng(71+int(f)*53+j*7+draw).permutation(len(counts))
                        mass=np.cumsum(counts[order])
                        for level in (128,512,"full"):
                            last=len(counts) if level=="full" else np.argmin(abs(mass-level))+1
                            src=t[np.isin(group[t],unique[order[:last]])]
                            dst=weighted.setdefault(str(level),np.zeros((3,len(y))))
                            if level=="full" and draw:
                                dst[draw,q]=dst[0,q]
                                continue
                            qs=np.r_[q,q]
                            z=np.c_[x[qs],np.repeat([-1,1],len(q))]
                            lo,hi=np.split(fit(src,[train_problem],z,off[qs])>.5,2)
                            prob=np.where(miss[q],reliability,1-reliability)
                            dst[draw,q]=unlocked[q]*(lo*(1-prob)+hi*prob)
            rate={}
            for level,draws in weighted.items():
                acts=draws.mean(0)
                fixed=float(acts[miss].sum())
                harmed=float(acts[~miss].sum())
                rate[level]=dict(expected_accuracy=float((np.sum(ref==y)+fixed-harmed)/len(y)),
                                 expected_fixed=fixed,expected_harmed=harmed)
            report[str(reliability)]=rate
        output[name]=report
    write(REPORT/"signal_control.json",output)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
