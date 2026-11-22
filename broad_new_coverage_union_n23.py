"""Test the new error coverage beyond 50 archived models and the gain on unseen bookings."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import panel, predictions
from broad_bill_exception_n23 import load, strata
from broad_booking_assignment_n23 import REPORT
from broad_booking_policy_n23 import ACTIONS
from broad_partition_evidence_n23 import ART, EXTRA, SOURCES, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development

SPECS = {
    "booking_assignment": ["own","assignment","means"],
    "learned_reference_weights": ["own","peers"],
    "spend_reference_network": ["pop_refs","balanced_refs","balanced_direct"],
    "solo_structure": ["own","hmm","window"],
}


def collect(row,y,base,seed,fold):
    native={}
    for folder,keys in SPECS.items():
        with np.load(ART/f"nov23_{folder}/{seed}.npz") as cache:
            for k,v in (("row",row),("y",y),("fold",fold),("base",base)):
                np.testing.assert_array_equal(cache[k],v)
            for k in keys:
                native[f"{folder}/{k}"]=cache[k]
    return native


def combination(old,native,base,fold,group,bill,size,y):
    def odds(p):
        return logit(p.clip(.002,.998))
    bank=np.column_stack([odds(base),odds(old)/np.sqrt(old.shape[1])])
    control=np.column_stack([odds(native[k]) for k in ("booking_assignment/own","learned_reference_weights/own",
                                   "spend_reference_network/pop_refs","solo_structure/own")])
    x=np.c_[bank,control/2]
    added=np.column_stack([odds(native[a])-odds(native[b]) for a,b in (
        ("booking_assignment/assignment","booking_assignment/own"),("booking_assignment/means","booking_assignment/own"),
        ("learned_reference_weights/peers","learned_reference_weights/own"),
        ("spend_reference_network/balanced_refs","spend_reference_network/pop_refs"),
        ("spend_reference_network/balanced_direct","spend_reference_network/pop_refs"),
        ("solo_structure/hmm","solo_structure/own"),("solo_structure/window","solo_structure/own"))])
    out={k:np.full(len(y),np.nan) for k in ("old","control","combined","shuffled")}
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=group[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(group[t],group[q]).size
            # Separate new evidence from the part linearly predictable from
            # existing models. Fit this residualization on training bookings.
            fit=Ridge(alpha=8.).fit(x[t],added[t])
            novel=(added-fit.predict(x))/np.sqrt(added.shape[1])
            extended=np.c_[novel,novel*(size>1)[:,None]]
            bins=np.digitize(base[t],np.quantile(base[t],np.arange(1,6)/6))
            bins=(bins*3+bill[t])*2+(size[t]>1)
            perm=np.arange(len(t));rng=np.random.default_rng(629)
            for cell in np.unique(bins):
                ix=np.flatnonzero(bins==cell)
                perm[ix]=rng.permutation(ix)
            for k,tr,te in (("old",bank[t],bank[q]),("control",x[t],x[q]),
                            ("combined",np.c_[x[t],extended[t]],np.c_[x[q],extended[q]]),
                            ("shuffled",np.c_[x[t],extended[t][perm]],np.c_[x[q],extended[q]])):
                out[k][q]=LogisticRegression(C=.5,max_iter=500).fit(tr,y[t]).predict_proba(te)[:,1]
    assert all(np.isfinite(v).all() for v in out.values())
    return out


def run():
    row,y,base,old,_,_,_=panel()
    ids,labels,frame,g,_=load()
    np.testing.assert_array_equal(ids,row)
    np.testing.assert_array_equal(labels,y)
    labels,release,m=development()
    np.testing.assert_array_equal(labels,y)
    miss=np.logical_and.reduce([p!=y for p in release.values()])
    bill=strata(frame)
    size=frame.BookingSize.to_numpy()
    core=miss&(((bill==2)&y)|((bill==0)&~y))
    masks={"miss":miss,"core":core,"core_solo":core&(size==1),"core_party":core&(size>1),"repeat":m["repeat"]}
    all_old=[];all_new=[];methods={}
    for i,seed in enumerate(SEEDS):
        _,extra,fold=predictions(row,y,seed,{**EXTRA,**dict(list(SOURCES.items())[:4])})
        archive=np.column_stack([old[i],extra])
        assert archive.shape==(len(y),50)
        all_old.append((archive>=.5)==y[:,None])
        anchor,direct,direct_fold=predictions(row,y,seed,ACTIONS)
        np.testing.assert_array_equal(anchor,base[i]);np.testing.assert_array_equal(fold,direct_fold)
        native=collect(row,y,base[i],seed,fold)
        all_new.append((np.column_stack(list(native.values()))>=.5)==y[:,None])
        for k,p in combination(direct,native,base[i],fold,g,bill,size,y).items():
            methods.setdefault(k,[]).append((p>=.5)==y)
    old_hits=np.stack(all_old);new_hits=np.stack(all_new)
    old_stable=old_hits.all(0).any(1);old_never=~old_hits.any((0,2))
    new_stable=new_hits.all(0)
    # Full-population cost of acting on dissent. 'unanimous' is observable
    # from the 50 hard decisions and the baseline, independent of correctness.
    dissent_cost = []
    for i,seed in enumerate(SEEDS):
        ref_hit=(base[i]>=.5)==y
        unanimous=(old_hits[i]==ref_hit[:,None]).all(1)
        nominated=(new_hits[i]!=ref_hit[:,None]).any(1)
        scope=unanimous&nominated
        dissent_cost.append(dict(seed=int(seed),unanimous=int(unanimous.sum()),nominated=int(scope.sum()),
                fixed=int((scope&~ref_hit).sum()),harmed=int((scope&ref_hit).sum()),
                core_fixes=int((core&scope&~ref_hit).sum())))
    coverage={}
    columns=list(native)
    for k,mask in masks.items():
        stat={}
        for col,h in zip(["any_new",*columns],[new_stable.any(1),*new_stable.T]):
            stat[col]=dict(stable=int((mask&h).sum()),added_beyond_old=int((mask&h&~old_stable).sum()),
                           previously_never_correct=int((mask&h&old_never).sum()))
        coverage[k]=dict(total=int(mask.sum()),old_stable=int((mask&old_stable).sum()),
                         old_never=int((mask&old_never).sum()),new=stat)
    assert coverage["core"]["old_stable"]==236 and coverage["miss"]["old_stable"]==465
    results={k:summarize(np.stack(v),(base>=.5)==y,masks) for k,v in methods.items()}
    write(REPORT/"combined_coverage.json",dict(coverage=coverage,combinations=results,unanimous_dissent=dissent_cost))
    print({k:v["new"]["any_new"] for k,v in coverage.items()},results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
