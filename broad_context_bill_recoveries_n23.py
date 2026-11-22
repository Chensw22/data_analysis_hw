"""Cross-check people rescued by two new self-only representations and their joint votes."""
import numpy as np
from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from broad_booking_policy_n23 import EXTRA
from relational_error_corrector import write


def run():
    row,y,base,older,_,names,_=panel()
    added=[predictions(row,y,s,EXTRA) for s in SEEDS]
    history=np.concatenate([older,np.stack([p for _,p,_ in added])],axis=2)
    old_ok=(history>=.5)==y[None,:,None]
    before=(base>=.5)==y
    repeat=~before.any(0)
    no_stable=repeat&~old_ok.all(0).any(1)
    no_hit=repeat&~old_ok.any((0,2))
    hits={}
    for i,seed in enumerate(SEEDS):
        fold=added[i][2]
        def read(source,key):
            with np.load(ART/f"nov23_{source}/{seed}.npz") as z:
                for field,value in [("row",row),("y",y),("fold",fold)]:
                    np.testing.assert_array_equal(z[field],value)
                return z[key]
        shape=read("broad_shared_passenger_modes","real_only")
        bills=read("broad_quantile_forest","standalone")
        own=read("broad_shared_passenger_modes","own")
        augmented=read("broad_shared_passenger_modes","real_modes")
        pred=base[i]>=.5
        dissent=((shape>=.5)==(bills>=.5))&((bills>=.5)!=pred)
        decisions={"shared_features_only":shape>=.5,"bills_only":bills>=.5,
                   "both_dissent":np.where(dissent,~pred,pred),
                   "own_and_bills":np.where((own>=.5)==(bills>=.5),bills>=.5,pred),
                   "augmented_and_bills":np.where((augmented>=.5)==(bills>=.5),bills>=.5,pred)}
        # Use the identical count and reversal direction per first-stage fold.
        # This is a comparison of which passengers were picked, without labels.
        matched=pred.copy()
        for k in np.unique(fold):
            for side in (False,True):
                candidates=np.flatnonzero((fold==k)&(pred==side))
                n=dissent[candidates].sum()
                ordered=candidates[np.argsort(abs(base[i,candidates]-.5),kind="stable")]
                matched[ordered[:n]]=not side
        decisions["matched_count_risk"]=matched
        for name,d in decisions.items():
            hits.setdefault(name,[]).append(d==y)
    result={}
    for name,values in hits.items():
        ok=np.stack(values)
        result[name]=dict(accuracy=ok.mean(1).tolist(),fixed=(ok&~before).sum(1).tolist(),
                          harmed=(~ok&before).sum(1).tolist(),ever_stable=int((~before.all(0)&ok.all(0)).sum()),
                          repeat_stable=int((repeat&ok.all(0)).sum()),new_stable=int((no_stable&ok.all(0)).sum()),
                          shared212_once=int((no_hit&ok.any(0)).sum()),shared212_stable=int((no_hit&ok.all(0)).sum()))
    s,b=[np.stack(hits[k]).all(0) for k in ("shared_features_only","bills_only")]
    overlap={}
    for scope,m in [("repeat",repeat),("historically_no_stable",no_stable)]:
        overlap[scope]=dict(only_shared=int((m&s&~b).sum()),only_bills=int((m&~s&b).sum()),
                            both=int((m&s&b).sum()),union=int((m&(s|b)).sum()))
    write(ROOT/"reports/nov23_broad_shared_passenger_modes/joint.json",dict(archive=len(names)+len(EXTRA),
          results=result,overlap=overlap))
    print(result["both_dissent"],overlap)


if __name__=="__main__":
    run()
