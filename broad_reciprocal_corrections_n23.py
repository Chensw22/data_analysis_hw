"""Require every new positive decision to replace an incumbent positive in the same roster."""
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel
from broad_bill_exception_n23 import load,strata
from broad_error_scope_balance_n23 import archived_votes
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import REPORT,scopes


def exchange(odds,base,cell,ids):
    blends=np.linspace(0,1,5)
    candidates=np.empty((len(ids),len(blends)),bool)
    rng=np.random.default_rng(892)
    for c in np.unique(cell[ids]):
        local=rng.permutation(np.flatnonzero(cell[ids]==c))
        rows=ids[local]
        n=int((base[rows]>=.5).sum())
        ranks=rankdata(odds[rows],axis=0)
        ranks=ranks/ranks.shape[0]
        direction=ranks[:,1]-ranks[:,0]
        for k,w in enumerate(blends):
            scores=ranks[:,0]+w*direction
            order=np.argsort(-scores,kind="stable")
            candidates[local,k]=False
            candidates[local[order[:n]],k]=True
        assert (candidates[local].sum(0)==n).all()
    np.testing.assert_array_equal(candidates[:,0],base[ids]>=.5)
    return candidates


def main():
    row,y,frame,g,_=load()
    a,b,bases,archive,*_=panel()
    np.testing.assert_array_equal(a,row); np.testing.assert_array_equal(b,y)
    old=(archived_votes(row,y,bases,archive)==y[None,:,None]).all(0).any(1)
    masks=scopes(y,frame)
    scope=strata(frame)*3+np.unique(frame.CryoSleep,return_inverse=True)[1]
    history,refs={},[]
    for i,s in enumerate(SEEDS):
        with np.load(ART/f"nov23_circuit_conditional_fit/{s}.npz") as z:
            base,fold=z["base"],z["fold"]
            np.testing.assert_array_equal(z["row"],row);np.testing.assert_array_equal(z["y"],y)
            np.testing.assert_array_equal(base,bases[i])
            gen=z["generative"];cond=z["conditional"]
        ref=(base>=.5)==y
        refs.append(ref)
        history.setdefault("native",[]).append((gen>=.5)==y)
        outputs={}
        for f in np.unique(fold):
            ix=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                t,q=ix[a],ix[b]
                assert not np.intersect1d(g[t],g[q]).size
                for model,source in (("generative",gen),("conditional",cond)):
                    scores=np.c_[base,source]
                    for title,cells in (("whole",np.zeros(len(y),int)),("billing",scope)):
                        train=exchange(scores,base,cells,t)
                        test=exchange(scores,base,cells,q)
                        winner=np.argmax((train==y[t,None]).sum(0))
                        for policy,k in (("direct",4),("learned",winner)):
                            out=outputs.setdefault(f"{model}/{title}/{policy}",np.empty(len(y),bool))
                            out[q]=test[:,k]
        for k,v in outputs.items():
            assert v.sum()==(base>=.5).sum()
            history.setdefault(k,[]).append(v==y)
    novel=np.logical_and.reduce(history["native"])&masks["repeat"]&~old
    assert novel.sum()==91
    stats={k:summarize(np.stack(h),np.stack(refs),{**masks,"original91":novel})
           for k,h in history.items()}
    write(REPORT/"exchanges.json",stats)
    for k,v in stats.items():print(k,v["accuracy"],v["stable"]["original91"],flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        main()
