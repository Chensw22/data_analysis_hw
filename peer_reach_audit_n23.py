"""Measure companion availability and the extra value of known companions over matched substitutes."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_matched_companions_n23 import BLOCKS
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, EXTRA, SOURCES, write
from released_correction_audit_n23 import development

REPORT = ROOT / "reports/nov23_peer_reach"


def topology(frame):
    party = frame.PassengerId.str.split("_").str[0]
    size = party.map(party.value_counts()).to_numpy()
    cabins = frame.Cabin.map(frame.Cabin.value_counts()).fillna(0).to_numpy()
    assert frame.dropna(subset=["Cabin"]).groupby("Cabin").PassengerId.apply(
        lambda s:s.str.split("_").str[0].nunique()).max()==1
    same = cabins>1
    assert np.all(~same | (size>1))
    return {"solo":size==1,"party_same_cabin":same,"party_no_observed_cabin_mate":(size>1)&~same}


def counts(mask, kind):
    result = {k:int((mask & v).sum()) for k,v in kind.items()}
    assert sum(result.values())==mask.sum()
    return result


def odds(p):
    return logit(np.clip(p,.002,.998))


def update(offset,x,y,t,q):
    def loss(beta):
        eta = offset[t]+x[t]@beta
        return np.sum(np.logaddexp(0,eta)-y[t]*eta)+6*(beta@beta), x[t].T@(expit(eta)-y[t])+12*beta
    fit = minimize(loss,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B")
    assert fit.success
    return expit(offset[q]+x[q]@fit.x)


def compare(a,b,y,mask):
    hit = a==y
    gain = hit.astype(int)-(b==y)
    return {"fixed":int(((gain>0)&mask).sum()),"harmed":int(((gain<0)&mask).sum()),
            "correct":int((hit&mask).sum())}


def main():
    rows,y,frame,groups,_ = load()
    labels,deployed,_ = development()
    np.testing.assert_array_equal(labels,y)
    missed = np.logical_and.reduce([v!=y for v in deployed.values()])
    paid = missed & y & (strata(frame)==2)
    zero = missed & ~y & (strata(frame)==0)
    masks = {"all":np.ones(len(y),bool),"941":missed,"696":paid|zero,"415":paid,"281":zero}
    train,test = [pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId","Cabin"]) for s in ("train","test")]
    gp = lambda f:f.PassengerId.str.split("_").str[0]
    assert not set(gp(train)).intersection(gp(test))
    assert not set(train.Cabin.dropna()).intersection(test.Cabin.dropna())
    topology_train,topology_test = topology(train),topology(test)
    coverage = {"official_train":counts(np.ones(len(train),bool),topology_train),
                "official_test":counts(np.ones(len(test),bool),topology_test)}
    kind = {k:v[rows] for k,v in topology_train.items()}
    coverage.update({k:counts(m,kind) for k,m in masks.items()})
    rr,yy,base,old,*_ = panel()
    np.testing.assert_array_equal(rr,rows)
    np.testing.assert_array_equal(yy,y)
    repeated = ((base>=.5)!=y).all(0)
    coverage["repeated_base_errors"] = counts(repeated,kind)
    members = {**EXTRA,**dict(list(SOURCES.items())[:4])}
    scores,stability,targets = {},{},{}
    for i,seed in enumerate(SEEDS):
        _,more,fold = predictions(rows,y,seed,members)
        votes = np.column_stack([old[i],more])>=.5
        assert votes.shape==(len(y),50)
        _,key = np.unique(np.c_[votes,base[i]>=.5,strata(frame)],axis=0,return_inverse=True)
        positive = np.bincount(key,weights=y)
        negative = np.bincount(key,weights=~y)
        opposite = np.where(y,negative[key],positive[key])>0
        p = base[i]
        ref = p>=.5
        scopes = {**masks,"core_collision":opposite&masks["696"],"baseline_error":ref!=y}
        coverage[str(seed)] = {k:counts(m,kind) for k,m in scopes.items() if k not in masks}
        remaining = int(np.floor(.85*len(y))+1-(ref==y).sum())
        available = int(((ref!=y)&~kind["solo"]).sum())
        targets[str(seed)] = dict(required_net=remaining, accompanied_errors=available,
                                  minimum_fraction=remaining/available)
        with np.load(ART/f"nov23_broad_matched_companions/{seed}.npz") as z:
            for k,v in (("row",rows),("y",y),("fold",fold),("base",p)):
                np.testing.assert_array_equal(z[k],v)
            diff = np.column_stack([odds(z["real"])-odds(z["matched_mean"]),
                                    *[odds(z["real"])-odds(z["replace_"+k]) for k in BLOCKS]])
            real,matched,transferred = z["real"]>=.5,z["matched_mean"]>=.5,z["transfer"]>=.5
        np.testing.assert_array_equal(real[kind["solo"]],matched[kind["solo"]])
        learned,sham = p.copy(),p.copy()
        for f in np.unique(fold):
            ids = np.flatnonzero((fold==f)&~kind["solo"])
            for a,b in GroupKFold(3,shuffle=True,random_state=713).split(ids,groups=groups[ids]):
                t,q = ids[a],ids[b]
                assert not np.intersect1d(groups[t],groups[q]).size
                learned[q] = update(odds(p),diff,y,t,q)
                # Preserve score direction, billing state, booking size and baseline
                # risk band when shuffling training contrasts among accompanied people.
                cells = np.c_[strata(frame)[t],np.minimum(frame.BookingSize[t],4),np.digitize(p[t],[.2,.4,.5,.6,.8])]
                _,inv = np.unique(cells,axis=0,return_inverse=True)
                rng = np.random.default_rng(936)
                permuted = diff.copy()
                for c in np.unique(inv):
                    member = t[inv==c]
                    permuted[member] = diff[rng.permutation(member)]
                sham[q] = update(odds(p),permuted,y,t,q)
        outputs = {"real_vs_matched":(real,matched),"fixed_transfer":(transferred,ref),
                   "learned_transfer":(learned>=.5,ref),"shuffled_transfer":(sham>=.5,ref),"base":(ref,ref)}
        scores[str(seed)] = {}
        for name,(dec,before) in outputs.items():
            scores[str(seed)][name] = {k:compare(dec,before,y,m) for k,m in scopes.items()}
            stability.setdefault(name,[]).append(dec==y)
    stable = {k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in masks.items()} for k,v in stability.items()}
    write(REPORT/"results.json",dict(coverage=coverage,plans=scores,stable=stable,local_85_requirement=targets))
    print(coverage, stable)
    for s,variants in scores.items():
        print(s,[(k,v["all"]["fixed"],v["all"]["harmed"],v["core_collision"]["fixed"],v["core_collision"]["harmed"]) for k,v in variants.items()])


if __name__=="__main__":
    with threadpool_limits(2):
        main()
