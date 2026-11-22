"""Measure how much unseen error coverage follows from a model's tendency to reverse decisions."""
import numpy as np
from archive_error_recovery_n23 import panel,predictions
from broad_booking_policy_n23 import EXTRA
from broad_diverse_rescue_n23 import MORE,REPORT,SEEDS
from broad_error_placement_control_n23 import SOURCES
from broad_partition_evidence_n23 import historical
from relational_error_corrector import write


def run():
    row,y,masks=historical()
    orig,target,base,old,*_=panel()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(target,y)
    prior={**EXTRA,**dict(list(SOURCES.items())[:4])}
    extra=np.stack([predictions(row,y,s,prior)[1] for s in SEEDS])
    prior_votes=np.concatenate([old,extra],axis=2)>=.5
    votes=np.stack([predictions(row,y,s,MORE)[1] for s in SEEDS])>=.5
    ref=base>=.5
    ref_ok=ref==y
    hits=votes==y[None,:,None]
    same=(hits.mean(1)>=ref_ok.mean(1)[:,None]-.01).all(0)
    hypothetical=list(MORE).index("hypothetical_outcome")
    agree=~(prior_votes!=ref[:,:,None]).all(0).any(1)
    never_oppose=~(prior_votes!=ref[:,:,None]).any((0,2))
    sure=(ref==ref[0]).all(0)
    # Recover these diagnostic cohorts by their observable historical vote
    # signatures first, and compare their correct and incorrect counterparts.
    group=agree&sure
    assert np.array_equal(group&~ref_ok.any(0),masks["unresolved"])
    assert np.array_equal(never_oppose&sure&~ref_ok.any(0),masks["never"])
    confidence=np.maximum(base,1-base).mean(0)
    code=(ref*np.array([1,2,4])[:,None]).sum(0)
    # Use narrow risk bins, exact broad voting pattern and the proportion of
    # original votes in dissent. No labels enter the cell assignments.
    dissent=(prior_votes!=ref[:,:,None]).mean((0,2))
    code=(((code*10+np.digitize(confidence,np.quantile(confidence,np.arange(.1,1,.1))))*2+agree)*2+never_oppose)*4+np.digitize(dissent,[1e-9,.05,.2])
    cells=[np.flatnonzero(code==c) for c in np.unique(code)]
    indices=np.arange(len(y))
    sets={"all_added":np.ones(len(MORE),bool),"without_error_audit":np.arange(len(MORE))!=hypothetical,
          "comparable":same,"error_audit":np.arange(len(MORE))==hypothetical}
    targets=[masks["unresolved"],masks["never"],group&ref_ok.all(0),never_oppose&sure&ref_ok.all(0)]
    labels=["new_stable_509","new_stable_172","could_wrongly_reverse_correct_4036","could_wrongly_reverse_unanimous_correct"]
    assert targets[2].sum()==4036
    flags={k:(votes[:,:,v]!=ref[:,:,None]).all(0).any(1) for k,v in sets.items()}
    draws={k:[] for k in sets}
    rng=np.random.default_rng(948)
    for _ in range(800):
        perm=indices.copy()
        for cell in cells:
            perm[cell]=rng.permutation(cell)
        for k,flag in flags.items():
            draws[k].append([int((flag[perm]&m).sum()) for m in targets])
    result={}
    for k,flag in flags.items():
        samples=np.array(draws[k])
        observed=[(flag&m).sum() for m in targets]
        result[k]={n:dict(observed=int(v),shuffled_mean=float(s.mean()),shuffled_95=np.quantile(s,[.025,.975]).tolist())
                   for n,v,s in zip(labels,observed,samples.T)}
    single=hits[:,:,hypothetical]
    write(REPORT/"volume_control.json",dict(results=result,matched_cells=len(cells),draws=800,
            audit_accuracy=single.mean(1).tolist(),audit_fixed=(single&~ref_ok).sum(1).tolist(),
            audit_harmed=(~single&ref_ok).sum(1).tolist(),opposed_fraction=(single!=ref_ok).mean(1).tolist()))


if __name__=="__main__":
    run()
