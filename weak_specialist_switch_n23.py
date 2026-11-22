"""Learn when to trust multiple weaker first-stage models on a disagreeing booking."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART,ROOT,SEEDS,panel,predictions
from relational_error_corrector import write

SPECIALISTS = {
    "registry":("bill_matched_register","register"), "budget":("simplex_likelihood","budget_prior"),
    "density":("manifest_tree","manifest_tree"),"registration_rank":("registration_rank","matched_registration"),
    "field_interactions":("field_factors","order3"),"input_projections":("projection_shapes","projection_shapes"),
    "local_gradient":("group_kernel","learned_twice"),
}


def estimate(base,opinions,y,t,q,specialists):
    negative=base<.5
    predicted=~negative
    error=predicted!=y
    opposite=np.where(negative[:,None],opinions,1-opinions)
    dissent=opposite>=.5
    bands=np.column_stack([np.minimum(base,1-base),dissent.mean(1),opposite.mean(1),
                           np.quantile(opposite,.75,axis=1)])
    if specialists:
        bands=np.column_stack([bands,opposite])
    scores=base[q].copy()
    for direction in (False,True):
        # Learn which disagreeing votes indicate a reliable reversal, using other
        # bookings from this outer fold. Unopposed cases keep their base score.
        eligible=(predicted==direction)&dissent.any(1)
        train=t[eligible[t]]
        query=np.flatnonzero(eligible[q])
        if not len(query) or len(train)<60:
            continue
        tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=50,random_state=707)
        tree.fit(bands[train],error[train])
        tn,qn=tree.apply(bands[train]),tree.apply(bands[q[query]])
        for node in np.unique(qn):
            calibration=train[tn==node]
            selected=query[qn==node]
            prior=np.mean(np.minimum(base[calibration],1-base[calibration]))
            prob=(error[calibration].sum()+16*prior)/(len(calibration)+16)
            # Transfer the excess error odds in this leaf, preserving the
            # passenger's original relative risk within it.
            gain=logit(np.clip(prob,.001,.999))-logit(np.clip(prior,.001,.999))
            r=expit(logit(np.minimum(base[q[selected]],1-base[q[selected]]).clip(.001,.999))+gain)
            scores[selected]=1-r if direction else r
    return scores


def run():
    row,y,base_all,bank,hard,_,_=panel()
    reference=(base_all>=.5)==y
    prev=(bank>=.5)==y[None,:,None]
    similar=(prev.mean(axis=1)>=reference.mean(1)[:,None]-.01).all(0)
    only_weak=~reference.any(0)&prev.all(0).any(1)&~prev.all(0)[:,similar].any(1)
    assert only_weak.sum()==335
    group=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0].to_numpy()[row]
    result={}
    out=ART/"nov23_weak_specialist_switch"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        base,opinions,fold=predictions(row,y,seed,SPECIALISTS)
        scores={k:np.full(len(y),np.nan) for k in ("pooled","specialist_id","shuffled")}
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=876).split(outer,groups=group[outer]):
                t,q=outer[a],outer[b]
                assert not np.intersect1d(group[t],group[q]).size
                shuffled=opinions.copy()
                # Permute whole opinion vectors within reference-risk bands and
                # within calibration/evaluation separately, without using outcomes.
                for part in (t,q):
                    band=(20*base[part]).astype(int)
                    for level in np.unique(band):
                        idx=part[band==level]
                        shuffled[idx]=opinions[np.random.default_rng(334+int(level)).permutation(idx)]
                for name,member in [("pooled",opinions),("specialist_id",opinions),("shuffled",shuffled)]:
                    scores[name][q]=estimate(base,member,y,t,q,name!="pooled")
        result[seed]={}
        for name,p in scores.items():
            assert np.isfinite(p).all()
            hit=(p>=.5)==y
            old=(base>=.5)==y
            result[seed][name]=dict(accuracy=float(hit.mean()),fixed=int((hit&~old).sum()),harmed=int((~hit&old).sum()),
                fixed_in_weak335=int(hit[only_weak].sum()),fixed_in_hard365=int(hit[hard].sum()),
                corrected_false_positives=int((hit&~old&~y).sum()),corrected_false_negatives=int((hit&~old&y).sum()))
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**scores)
        print(seed,result[seed],flush=True)
    write(ROOT/"reports/nov23_weak_specialist_switch/result.json",dict(members=SPECIALISTS,results=result))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
