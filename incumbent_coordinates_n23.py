"""Replay learned position corrections on the public champion's original validation rows."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT
from paired_cabin_coordinates_n23 import coordinate_maps, correct, features
from relational_error_corrector import write


def run():
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        row,y,fold,champ,prob,unseen=(z[k] for k in ("row","y","fold","v131","v108","unseen"))
    historical=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
    for k,target in [("row",row),("true",y),("fold",fold),("v131",champ)]:
        np.testing.assert_array_equal(historical[k],target)
    full=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in {"Name","Transported"})
                    for s in ("train","test")],ignore_index=True)
    coordinates,_=coordinate_maps(full)
    raw=full.iloc[row]
    np.testing.assert_array_equal(raw.PassengerId,historical.PassengerId)
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    bank={k:features(raw,s[row]) for k,s in coordinates.items()}
    bank["registration"]=features(raw,None)
    override=champ!=(prob>=.5)
    pred={k:np.full(len(y),np.nan) for k in bank}
    for f in np.unique(fold):
        held=np.flatnonzero(fold==f)
        np.testing.assert_array_equal(unseen[held],~np.isin(group[held],group[fold!=f]))
        for a,b in GroupKFold(3,shuffle=True,random_state=705).split(held,groups=group[held]):
            t,q=held[a],held[b]
            # Deployment consists of new bookings. Learn the adjustment only on
            # rows with that status, and where v131 permits a score-based decision.
            t=t[unseen[t]&~override[t]]
            assert not np.intersect1d(group[t],group[q]).size
            for k,x in bank.items():
                pred[k][q]=correct(x,prob,y,t,q)
    report={};cache={}
    ref=champ==y
    for name,p in pred.items():
        assert np.isfinite(p).all()
        choice=np.where(override,champ,p>=.5)
        cache[name]=choice
        ok=choice==y
        scores={}
        for k,m in [("whole",np.ones(len(y),bool)),("new_booking",unseen)]:
            scores[k]=dict(n=int(m.sum()),accuracy=float(ok[m].mean()),base_accuracy=float(ref[m].mean()),
                           fixed=int((m&ok&~ref).sum()),harmed=int((m&ref&~ok).sum()),
                           net_by_fold=[int((m&(fold==f)&ok).sum()-(m&(fold==f)&ref).sum()) for f in np.unique(fold)])
        report[name]=scores
    out=ART/"nov23_incumbent_coordinates"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/"oof.npz",row=row,y=y,fold=fold,v131=champ,unseen=unseen,**cache)
    write(ROOT/"reports/nov23_incumbent_coordinates/result.json",dict(protected=int(override.sum()),scores=report))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
