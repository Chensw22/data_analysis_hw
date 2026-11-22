"""Can other independent predictions identify when CARTE corrects the strong reference?"""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,predictions
from relational_error_corrector import write


MEMBERS={"carte":("carte","p"),"tabpfn":("weighted_tabpfn_manifest","learned"),
         "spatial":("shared_booking_layout","teacher_with_links"),
         "distilled":("training_companion_outcomes","companion_distilled")}


def run():
    with np.load(ART/f"nov23_group_kernel/{SEEDS[0]}.npz") as z:
        row,y=z["row"],z["y"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","HomePlanet"]).iloc[row]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    earth=raw.HomePlanet.eq("Earth").to_numpy()
    bundle=[predictions(row,y,s,MEMBERS) for s in SEEDS]
    repeat=np.logical_and.reduce([(b>=.5)!=y for b,_,_ in bundle])
    report={}
    all_hits={}
    out=ART/"nov23_carte_disagreement_evidence"
    out.mkdir(exist_ok=True)
    for seed,(base,bank,fold) in zip(SEEDS,bundle):
        ref=base>=.5; expert=bank[:,0]>=.5
        disputed=expert!=ref
        polarity=2*expert.astype(float)-1
        # Positive values support CARTE's proposed direction. All inputs are
        # first-stage scores from models excluding every booking in this fold.
        logits=logit(np.column_stack([base,bank]).clip(.001,.999))*polarity[:,None]
        methods={"base_risk":[0],"two_models":[0,1],"all_evidence":[0,1,2,3,4]}
        decisions={k:ref.copy() for k in methods}
        matched={k:ref.copy() for k in methods}
        for f in np.unique(fold):
            rows=np.flatnonzero((fold==f)&disputed)
            for a,b in GroupKFold(3,shuffle=True,random_state=189).split(rows,groups=group[rows]):
                t,q=rows[a],rows[b]
                assert not np.intersect1d(group[t],group[q]).size
                for name,cols in methods.items():
                    x=np.column_stack([logits[:,cols],polarity])
                    reg=LogisticRegression(C=.15,max_iter=250).fit(x[t],expert[t]==y[t])
                    flip=reg.predict_proba(x[q])[:,1]>=.5
                    decisions[name][q[flip]]=expert[q[flip]]
                    # The comparison receives precisely the same number of
                    # changes per direction within each untouched subfold.
                    for label in (False,True):
                        candidates=q[expert[q]==label]
                        n=int((flip&(expert[q]==label)).sum())
                        chosen=candidates[np.argsort(abs(base[candidates]-.5),kind="stable")[:n]]
                        matched[name][chosen]=expert[chosen]
        with np.load(ART/f"nov23_identical_bills/{seed}.npz") as old:
            np.testing.assert_array_equal(row,old["row"])
            hard=earth&repeat&old["eligible"]
        before=ref==y
        scores={}
        for name,d in {**decisions,**{k+"_matched":v for k,v in matched.items()},"all_carte":expert}.items():
            ok=d==y
            scores[name]=dict(accuracy=float(ok.mean()),changes=int((d!=ref).sum()),fixed=int((ok&~before).sum()),
                              harmed=int((~ok&before).sum()),repeat=int((ok&repeat).sum()),earth430=int((ok&hard).sum()))
            all_hits.setdefault(name,[]).append(ok)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**decisions)
        report[seed]=scores
    stable={k:int((repeat&np.logical_and.reduce(v)).sum()) for k,v in all_hits.items()}
    write(ROOT/"reports/nov23_carte_disagreement_evidence/result.json",dict(results=report,stable1070=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
