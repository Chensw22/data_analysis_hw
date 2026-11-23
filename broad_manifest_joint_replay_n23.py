"""Combine billing and ship-wide exposure and replay correction on the actual v131 anchor."""
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from broad_attribution_transfer_n23 import tree_parts
from broad_explanation_pairs_n23 import held_model
from broad_explanation_stability_n23 import DOMAINS
from broad_geographic_transfer_n23 import load
from broad_manifest_exposure_n23 import stream_context
from broad_manifest_vector_ranks_n23 import process,REPORT
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_release_miss_coverage_n23 import summarize
from broad_twin_surfaces_n23 import scopes
from exaone_validation_data import compact_features
from released_correction_audit_n23 import development


def main():
    row,y,frame,g,_=load()
    masks=scopes(y,frame)
    roster=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Transported","Name"))
                      for s in ("train","test")],ignore_index=True)
    ranks=process(compact_features(roster))
    group,member,own,other,_,_,_=stream_context()
    np.testing.assert_array_equal(group[row],g)
    def extra(arr,prefix):
        arr=arr[row]
        return pd.DataFrame(arr,columns=[f"{prefix}{i}" for i in range(arr.shape[1])])
    blocks=[extra(ranks["marginal"],"marginal"),extra(ranks["joint"],"vector"),
            extra(np.c_[group,member,own],"booking"),extra(other,"other_stream")]
    joint=pd.concat([frame,*blocks],axis=1)
    dev_y,release,_=development()
    np.testing.assert_array_equal(y,dev_y)
    results,refs,hs={},[],{}
    for s in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{s}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        assert np.array_equal(ref,release[131] if s=="v131" else base>=.5)
        lock=ref!=(base>=.5)
        ref_hit=ref==y
        models={}
        for k,design in (("raw",frame),("joint",joint)):
            if k=="raw" and s!="v131":
                with np.load(ART/f"nov23_attribution_transfer/{s}.npz") as z:
                    for label,v in (("row",row),("y",y),("fold",fold)):np.testing.assert_array_equal(z[label],v)
                    tree,phi=z["tree"],z["phi"]
            else:
                phi,raw=tree_parts(design,y,g,fold)
                tree=expit(raw)
            fields=list(DOMAINS)
            if k=="joint":
                fields.extend([list(b.columns) for b in blocks])
            by_domain=np.column_stack([phi[:,design.columns.get_indexer(names)].sum(1) for names in fields])
            np.testing.assert_allclose(by_domain.sum(1),phi.sum(1),atol=1.e-8)
            pred={j:ref.copy() for j in ("risk","matched","shuffled")}
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&~lock)
                for j,p in held_model(ids,base,tree,by_domain,y,g).items():
                    pred[j][ids]=p[ids]>=.5
            models[k+"/tree"]=tree>=.5
            for j,v in pred.items():
                assert np.array_equal(v[lock],ref[lock])
                models[k+"/"+j]=v
        stats={}
        for k,p in models.items():
            h=p==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~ref_hit).sum()),harmed=int((~h&ref_hit).sum()),
                          net_by_fold=[int(h[fold==f].sum())-int(ref_hit[fold==f].sum()) for f in np.unique(fold)],
                          fixed_repeat=int((h&~ref_hit&masks["repeat"]).sum()))
            if s!="v131":hs.setdefault(k,[]).append(h)
        if s!="v131":refs.append(ref_hit)
        results[s]=dict(reference_accuracy=float(ref_hit.mean()),locked=int(lock.sum()),models=stats)
        print(s,[(k,v["fixed"],v["harmed"]) for k,v in stats.items() if k.endswith("matched")],flush=True)
    write(REPORT/"joint_replay.json",dict(splits=results,grouped={k:summarize(np.stack(h),np.stack(refs),masks)
                                                                for k,h in hs.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
