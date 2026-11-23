"""Check whether new stable recoveries persist after equalizing class-decision counts."""
import numpy as np
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel
from broad_bill_exception_n23 import load, strata
from broad_error_scope_balance_n23 import archived_probabilities
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_twin_surfaces_n23 import REPORT, scopes


def align(p, cell, decision, seed):
    # Label-free policy assigning the same number of positives as the
    # challenger in each cell. Record order cannot break probability ties.
    rng=np.random.default_rng(seed)
    result=np.zeros_like(p,dtype=bool)
    for c in np.unique(cell):
        idx=rng.permutation(np.flatnonzero(cell==c))
        count=int(decision[idx].sum())
        order=np.argsort(-p[idx],axis=0,kind="stable")[:count]
        result[idx[order],np.arange(p.shape[1])] = True
        assert (result[idx].sum(0)==count).all()
    return result


def main():
    rows,y,bases,archive,*_=panel()
    a,b,frame,_,_=load()
    np.testing.assert_array_equal(a,rows);np.testing.assert_array_equal(b,y)
    p=archived_probabilities(rows,y,bases,archive)
    default=(p>=.5)==y[None,:,None]
    stable=default.all(0).any(1)
    never=~default.any((0,2))
    m=scopes(y,frame)
    votes=[]
    folds=[]
    for i,s in enumerate(SEEDS):
        with np.load(ART/f"nov23_population_circuit/{s}.npz") as z:
            for k,v in (("row",rows),("y",y),("base",bases[i])):np.testing.assert_array_equal(z[k],v)
            votes.append(z["one_subtype"]>=.5);folds.append(z["fold"])
    gen=np.stack(votes)
    novel=(gen==y).all(0)&m["repeat"]&~stable
    assert novel.sum()==91 and (novel&never).sum()==11
    state=strata(frame)
    sleep=np.unique(frame.CryoSleep,return_inverse=True)[1]
    records={}
    for name in ("global","bill_sleep","risk_bill_sleep"):
        controls=[]
        for tie_seed in [271,863,997]:
            panels=[]
            for i,s in enumerate(SEEDS):
                # Rank passengers within the same first-stage model fit.
                f=folds[i]
                cell=f.copy()
                if name!="global":
                    cell=(cell*3+state)*(sleep.max()+1)+sleep
                if name=="risk_bill_sleep":
                    q=np.digitize(bases[i],np.quantile(bases[i],[.2,.4,.6,.8]))
                    cell=5*cell+q
                panels.append(align(p[i],cell,gen[i],tie_seed))
            h=np.stack(panels)==y[None,:,None]
            possible=h.all(0).any(1)
            counts=(h.all(0)&novel[:,None]).sum(0)
            controls.append(dict(still_novel=int((novel&~possible).sum()),
                still_novel_false=int((novel&~possible&~y).sum()),
                old_never_still_novel=int((novel&never&~possible).sum()),
                most_by_single=int(counts.max()),
                any_repeat_stable=int((m["repeat"]&possible).sum()),
                max_single_accuracy=h.mean(1).max(1).tolist()))
        records[name]=controls
    with np.load(ART/f"nov23_population_circuit/{SEEDS[0]}.npz") as z:
        assert z["one_subtype"].shape == y.shape
    write(REPORT/"operating_points.json",dict(initial=int(novel.sum()),never=int((novel&never).sum()),
            novel_false=int((novel&~y).sum()),generator_positive=gen.sum(1).tolist(),
            baseline_positive=(bases>=.5).sum(1).tolist(),results=records))
    print(records,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
