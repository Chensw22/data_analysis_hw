"""Compare never-stably-correct cases with passengers sharing the same kinds of model agreement."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_weak_disagreement_n23 import ROOT, ART, SEEDS, EXTRA, SOURCES, panel, predictions
from broad_geographic_transfer_n23 import BILLS, load
from broad_refit_dissent_n23 import features
from broad_weak_complement_n23 import matched_order
from relational_error_corrector import write

REPORT = ROOT / "reports/nov23_broad_unresolved_markers"
OUTPUT = ART / "nov23_broad_unresolved_markers"


def match_pattern(row,y,base,old):
    adds = {**EXTRA,**dict(list(SOURCES.items())[:4])}
    recent = np.stack([predictions(row,y,s,adds)[1] for s in SEEDS])
    opinions = np.concatenate([old,recent],axis=2)>=.5
    ref = base>=.5
    dissent = opinions!=ref[:,:,None]
    stable = (ref==ref[0]).all(0)
    occasional = dissent.any((0,2))
    scope = stable & ~dissent.all(0).any(1)
    case = scope & (ref[0]!=y)
    assert scope.sum()==4545 and case.sum()==509 and (case&~occasional).sum()==172
    return scope,case,occasional


def cells(base,width):
    return (base>=.5).astype(int)+2*np.floor(np.minimum(base,1-base)/width).astype(int)


def profiles(frame,groups,base,scope,case,occasional):
    cols = {"Earth":frame.HomePlanet.eq("Earth"),"sleep":frame.CryoSleep.eq("True"),
            "positive_spend":frame[BILLS].gt(0).any(axis=1),"multi":frame.BookingSize.gt(1),
            "young":frame.Age.lt(18),"missing_age_or_bill":frame[["Age",*BILLS]].isna().any(axis=1),
            "side_S":frame.Side.eq("S")}
    cols.update({f"positive_{c}":frame[c].gt(0) for c in BILLS})
    fields = np.asarray(list(cols.values()),float).T
    out = {}
    for k,p in enumerate(base):
        strata = 2*cells(p,.02)+occasional
        diffs,cases,controls = [],[],[]
        contributions=np.zeros_like(fields)
        eligible_cases=np.zeros(len(fields),bool)
        for j in np.unique(strata[scope]):
            a = np.flatnonzero(scope & case & (strata==j))
            b = np.flatnonzero(scope & ~case & (strata==j))
            if len(a) and len(b):
                mean_control=fields[b].mean(0)
                diffs.append(fields[a].mean(0)-mean_control)
                contributions[a]=fields[a]-mean_control
                contributions[b]=(mean_control-fields[b])*len(a)/len(b)
                eligible_cases[a]=True
                cases.append(len(a))
                controls.append(len(b))
        delta=np.average(diffs,axis=0,weights=cases)
        contributions[eligible_cases]-=delta
        by_booking=pd.DataFrame(contributions).groupby(groups).sum().to_numpy()
        se=np.sqrt((by_booking**2).sum(0))/sum(cases)
        out[str(SEEDS[k])] = dict(matched_cases=sum(cases), matched_controls=sum(controls), difference_pp=dict(zip(cols,100*delta)),
                                  cluster_ci95_pp={n:[float(100*(v-1.96*e)),float(100*(v+1.96*e))] for n,v,e in zip(cols,delta,se)})
    return out


def witness(x,base,error,a,b,permute):
    # A Gaussian-kernel mean difference estimates what observed fields are more
    # prevalent among errors than among equally confident correct predictions.
    z = np.clip((x-x[a].mean(0))/x[a].std(0).clip(.3),-5,5)
    rng=np.random.default_rng(733)
    omega=rng.normal(size=(z.shape[1],192))/np.sqrt(z.shape[1])
    phi=np.sqrt(2)*np.cos(z@omega+rng.uniform(0,2*np.pi,192))
    phi=np.column_stack([phi,z])
    block=cells(base,.05)
    evidence=np.zeros(len(x))
    for direction in (False,True):
        ids=a[(base[a]>=.5)==direction]
        difference=np.zeros(phi.shape[1])
        for j in np.unique(block[ids]):
            c=ids[block[ids]==j]
            e=error[c]
            if permute:
                e=rng.permutation(e)
            # Zero-sum residual weights exactly cancel between-cell prevalence.
            difference += (e-e.mean())@phi[c]
        evidence[(base>=.5)==direction] = phi[(base>=.5)==direction]@difference
    scale=evidence[b].std()
    return evidence/max(scale,.0001)


def run():
    row,y,base,old,_,_,_=panel()
    rows,truth,frame,groups,_=load()
    np.testing.assert_array_equal(row,rows)
    np.testing.assert_array_equal(y,truth)
    scope,case,occ=match_pattern(row,y,base,old)
    raw_distributions=profiles(frame,groups,base,scope,case,occ)
    x=features(frame)[0]
    masks=dict(ever=((base>=.5)!=y).any(0),repeat=((base>=.5)!=y).all(0),unresolved=case,
               sporadic=case&occ,unanimous=case&~occ,mirror=scope&~case)
    plans=[]
    for i,s in enumerate(SEEDS):
        with np.load(ART/f"nov23_group_kernel/{s}.npz") as z:
            plans.append((str(s),base[i],z["fold"],base[i]>=.5))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        for field,expected in (("row",row),("y",y)):
            np.testing.assert_array_equal(z[field],expected)
        plans.append(("v131",z["v108"],z["fold"],z["v131"]))
    result, history={},{}
    OUTPUT.mkdir(exist_ok=True)
    for name,p,fold,ref in plans:
        methods={k: np.full(len(y),np.nan) for k in ("confidence","kernel","shuffled")}
        error=(p>=.5)!=y
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            parts=[outer[j] for _,j in GroupKFold(3,shuffle=True,random_state=135).split(outer,groups=groups[outer])]
            for i,q in enumerate(parts):
                a,b=parts[(i+1)%3],parts[(i+2)%3]
                assert not set(groups[q]) & set(groups[np.r_[a,b]]) and not set(groups[a]) & set(groups[b])
                reps={k:[] for k in methods}
                for train,cal in ((a,b),(b,a)):
                    risk=np.column_stack([logit(np.minimum(p,1-p).clip(.001,.5)),p>=.5])
                    for key in methods:
                        design=risk
                        if key!="confidence":
                            w=witness(x,p,error,train,cal,key=="shuffled")
                            design=np.column_stack([risk,w])
                        model=LogisticRegression(C=.5).fit(design[cal],error[cal])
                        reps[key].append(model.predict_proba(design[q])[:,1])
                for key in methods:
                    methods[key][q]=np.mean(reps[key],axis=0)
        result[name]={}
        for key,risk in methods.items():
            assert np.isfinite(risk).all()
            prob=np.where(p>=.5,1-risk,risk)
            hit=np.where(ref!=(p>=.5),ref,prob>=.5)==y
            delta=hit.astype(int)-(ref==y)
            result[name][key]=dict(accuracy=float(hit.mean()),fixed=int((delta>0).sum()),harmed=int((delta<0).sum()),
                              net_folds=[int(delta[fold==j].sum()) for j in np.unique(fold)])
            if name!="v131":
                history.setdefault(key,[]).append(hit)
                result[name][key]["matched_order"]={label:matched_order(prob,p,scope&select,case)
                         for label,select in (("unanimous",~occ),("sporadic",occ))}
        np.savez_compressed(OUTPUT/f"{name}.npz",row=row,y=y,fold=fold,base=p,**methods)
    stable={k:{n:int((np.logical_and.reduce(h)&m).sum()) for n,m in masks.items()} for k,h in history.items()}
    write(REPORT/"result.json",dict(scope=int(scope.sum()),cases=int(case.sum()),unanimous_total=int((scope&~occ).sum()),
        case_by_direction={str(k):int((case&((base[0]>=.5)==k)).sum()) for k in (False,True)},
        profiles=raw_distributions,results=result,stable=stable))
    print(result["v131"],stable,flush=True)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
