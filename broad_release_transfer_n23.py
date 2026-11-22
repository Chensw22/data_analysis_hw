"""Compare matched local/public rankings using archived decisions and aggregate public scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau,spearmanr
from sklearn.metrics import log_loss
from broad_partition_evidence_n23 import ART,ROOT,historical,write
from exaone_validation_data import load_data,random_references
from three_family_mean_screen import cached_fold_scores

REPORT=ROOT/"reports/nov23_release_transfer"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched():
    official=json.loads((REPORT/"official.json").read_text())["scores"]
    earlier=json.loads((ROOT/"reports/common_error_followup/local_public_agreement.json").read_text())
    tiers=pd.read_csv(ROOT/"reports/method_tiers.csv").set_index("version")
    inputs=load_data()
    row,y=inputs["dev"],inputs["y"]
    splits,_,_=random_references(inputs)
    fold=np.empty(len(row),int)
    for k,(_,v) in enumerate(splits):fold[v]=k
    test=pd.read_csv(ROOT/"data/raw/test.csv",usecols=["PassengerId"])
    panel=[]
    for rec in earlier["versions_by_development"]:
        name=rec["version"]
        receipt=ROOT/rec["source"]
        released=json.loads(receipt.read_text())
        ref=str(released["submission_ref"])
        assert released["version"]==name and released["competition"]=="spaceship-titanic"
        assert released["evaluation"]["Status"]=="COMPLETE"
        score=float(official[ref])
        assert score==float(released["evaluation"]["Public Score"])
        csv=ART/released["run"]/"submission.csv"
        assert digest(csv)==released["file_sha256"]
        predictions=pd.read_csv(csv)
        np.testing.assert_array_equal(predictions.PassengerId,test.PassengerId)
        assert predictions.Transported.dtype==bool
        if name.startswith("v96_"):
            p=ART/"v96_oof_completion/oof.npz"
        elif name.startswith("v80_"):
            p=ART/"top5_crossfamily_phase/screen_20261112.npz"
        else:
            run=tiers.loc[name,"source_runs"].split(";")[0]
            p=ART/run/f"{name}_seed20261112.npz"
        with np.load(p,allow_pickle=False) as z:
            prob=z["prob"].astype(float)
            explicit=False
            for key in ("dev_ids","development"):
                if key in z:
                    np.testing.assert_array_equal(z[key],row)
                    explicit=True
            if "fold_id" in z:np.testing.assert_array_equal(z["fold_id"],fold)
            if "fold_accuracy" in z:cached_fold_scores(prob,z["fold_accuracy"],y,splits)
            assert explicit or "fold_accuracy" in z
        assert np.isclose(((prob>=.5)==y).mean(),rec["development"],rtol=0,atol=1e-12)
        if name in tiers.index:
            screen=pd.read_csv(ROOT/"reports"/tiers.loc[name,"source_runs"].split(";")[0]/"screen_results.csv")
            selected=screen[screen.version==name].iloc[0]
            assert abs(log_loss(y,prob)-float(selected.log_loss))<1e-6
        panel.append(dict(version=name,ref=ref,score=score,prob=prob,test=predictions.Transported.to_numpy(),
                          cached=str(p.relative_to(ROOT)),id_check="row IDs" if explicit else "three fold accuracies and log loss",
                          time=released["started_at_utc"],sha=digest(p)))
    return row,y,panel


def agreement(p,q):
    a,b=np.triu_indices(len(p),1)
    sign=np.sign(np.asarray(p)[a]-np.asarray(p)[b])
    actual=np.sign(np.asarray(q)[a]-np.asarray(q)[b])
    valid=(sign!=0)&(actual!=0)
    return dict(spearman=float(spearmanr(p,q).statistic),kendall=float(kendalltau(p,q).statistic),
                comparable=int(valid.sum()),confirmed=int((valid&(sign==actual)).sum()))


def transported_difference(a,b,ta,tb,y,group_train,group_test):
    # Evaluate only aggregate score differences between two released methods.
    # The success frequency of a disagreement direction is learned locally.
    delta=(a==y).astype(int)-(b==y)
    outcome=0.
    for s in (False,True):
        sel=(a!=b)&(a==s)
        target=(ta!=tb)&(ta==s)
        prior=delta[sel].mean() if sel.any() else 0.
        for c in np.unique(group_test[target]):
            # A ten-passenger prior pools sparse roster strata within the same direction.
            ix=sel&(group_train==c)
            outcome+=(delta[ix].sum()+10*prior)/(ix.sum()+10)*np.sum(target&(group_test==c))
    return float(outcome/len(ta))


def main():
    row,y,entries=matched()
    old,labels,masks=historical()
    np.testing.assert_array_equal(old,row);np.testing.assert_array_equal(labels,y)
    base=next(e for e in entries if e["version"].startswith("v71_"))
    baseline=base["prob"]>=.5
    dev=[];public=[];difficulty={k:[] for k in masks}
    results=[]
    for e in entries:
        ok=(e["prob"]>=.5)==y
        local=ok.mean()
        stats={k:int((mask&ok).sum()) for k,mask in masks.items()}
        for k,v in stats.items():difficulty[k].append(v)
        results.append({**{k:v for k,v in e.items() if k not in ("test","prob")}, "local":float(local),
                "correct_in_cohort":stats,"fixed_vs_v71":int((ok&(baseline!=y)).sum()),
                "harmed_vs_v71":int((~ok&(baseline==y)).sum())})
        dev.append(local);public.append(e["score"])
    summary={k:agreement(v,public) for k,v in dict(local=dev,**difficulty).items()}
    context=[]
    for s in ("train","test"):
        df=pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId","CryoSleep","HomePlanet","RoomService","Spa","ShoppingMall","VRDeck","FoodCourt"])
        g=df.PassengerId.str.split("_").str[0]
        paid=df[["Spa","VRDeck","ShoppingMall","FoodCourt","RoomService"]].gt(0).any(axis=1)
        context.append(np.char.add(np.char.add(df.HomePlanet.fillna("?").to_numpy(dtype=str),paid.to_numpy().astype(str)),
                                  np.minimum(g.map(g.value_counts()).to_numpy(),3).astype(str)))
    strata=context[0][row],context[1]
    comparisons=[]
    for i,a in enumerate(entries):
        for b in entries[:i]:
            av,bv=[v["prob"]>=.5 for v in (a,b)]
            at,bt=a["test"],b["test"]
            local=float(np.mean(av==y)-np.mean(bv==y))
            public_diff=a["score"]-b["score"]
            comparisons.append(dict(a=a["version"],b=b["version"],local=local,public=public_diff,
                direction=transported_difference(av,bv,at,bt,y,np.zeros(len(y)),np.zeros(len(at))),
                roster=transported_difference(av,bv,at,bt,y,*strata),
                disagreement_dev=float((av!=bv).mean()),disagreement_test=float((at!=bt).mean())))
    checks={}
    for key in ("local","direction","roster"):
        selected=[r for r in comparisons if r[key]!=0 and r["public"]!=0]
        checks[key]=dict(n=len(selected),ordering_matches=sum(np.sign(r[key])==np.sign(r["public"]) for r in selected).item(),
                mean_abs_gap_pp=float(100*np.mean([abs(r[key]-r["public"]) for r in comparisons])))
    write(REPORT/"early_models.json",dict(models=results,rankings=summary,pairwise=checks,comparisons=comparisons))
    print(summary,checks)


if __name__=="__main__":
    main()
