"""Compare complementary, diverse and accurate shortlists on entirely different bookings."""
from collections import Counter
import numpy as np
from scipy.linalg import qr
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import ART,ROOT,SEEDS,predictions
from broad_booking_policy_n23 import ACTIONS
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import historical
from broad_weak_complement_n23 import combine
from relational_error_corrector import write

# Only first-stage predictions whose entire outer evaluation fold was excluded
# from supervised fitting. Do not reuse combined / oracle / placebo columns.
MORE = {
    "flow":("conditional_bill_flow","flow"), "pathways":("multiple_pathways","symmetric"),
    "learned_training_weights":("booking_meta_weights","aligned"),
    "label_transition":("broad_noise_transition","adjusted"), "within_booking":("broad_conditional_counts","conditional"),
    "hypothetical_outcome":("broad_hypothetical_labels","direct_error_audits"),
    "gradient_projection":("competing_gradients","projection"),
    "companion_distillation":("training_companion_outcomes","companion_distilled"),
    "booking_layout":("shared_booking_layout","teacher_with_links"),
    "shortcut_alignment":("shortcut_alignment","aligned"),
    "high_loss_training":("broad_train_error_margins","robust"),
    "denoising":("broad_passenger_diffusion","direct"), "service_transport":("broad_service_transport","learned"),
    "partition_tree":("broad_partition_evidence","native"), "covariate_weighting":("broad_target_shift","natural_domain")
}
OUT = ART/"nov23_broad_diverse_rescue"
REPORT = ROOT/"reports/nov23_broad_diverse_rescue"


def shortlist(base,p,y,t):
    # All rankings use only the calibration bookings; no historical masks.
    ref=base[t]>=.5
    correct=(p[t]>=.5)==y[t,None]
    accurate=np.argsort(-correct.mean(0),kind="stable")[:6]
    missed=ref!=y[t]
    chosen=[]
    for _ in range(6):
        new=(correct & missed[:,None]).sum(0)
        new[chosen]=-1
        j=int(np.argmax(new))
        chosen.append(j)
        missed &= ~correct[:,j]
    d=logit(p[t].clip(.002,.998))-logit(base[t].clip(.002,.998))[:,None]
    d-=d.mean(0)
    _,_,pivots=qr(d,mode="economic",pivoting=True)
    return dict(accuracy=accurate, coverage=np.asarray(chosen), diversity=pivots[:6],
                all=np.arange(p.shape[1]), reference=np.arange(0))


def run():
    row,y,_,g,_=load()
    orig,target,scopes=historical()
    np.testing.assert_array_equal(orig,row)
    np.testing.assert_array_equal(target,y)
    methods={**ACTIONS,**MORE}
    keys=list(methods)
    assert len(methods)==len(set(methods.values()))
    OUT.mkdir(exist_ok=True)
    votes,baseline,history=[],[],{}
    freq={k:Counter() for k in ["accuracy","coverage","diversity"]}
    for seed in SEEDS:
        base,p,fold=predictions(row,y,seed,methods)
        votes.append(p>=.5)
        baseline.append(base>=.5)
        scores={}
        chosen_correct={}
        for k in np.unique(fold):
            v=np.flatnonzero(fold==k)
            for a,b in GroupKFold(3,shuffle=True,random_state=846).split(v,groups=g[v]):
                t,q=v[a],v[b]
                assert not np.intersect1d(g[t],g[q]).size
                for label,cols in shortlist(base,p,y,t).items():
                    vals=combine(base,p[:,cols],y,t,q)
                    scores.setdefault(label,np.full(len(y),np.nan))[q]=vals
                    if label in freq:
                        freq[label].update(keys[j] for j in cols)
                    if len(cols):
                        chosen_correct.setdefault(label,np.zeros(len(y),bool))[q]=((p[q][:,cols]>=.5)==y[q,None]).any(1)
        for label,score in scores.items():
            assert np.isfinite(score).all()
            history.setdefault(label,[]).append((score>=.5,chosen_correct.get(label,np.zeros(len(y),bool))))
        np.savez_compressed(OUT/f"{seed}.npz",row=row,y=y,base=base,fold=fold,**scores)
    ref=np.stack(baseline)==y
    candidates=np.stack(votes)==y[None,:,None]
    added=candidates[:,:,len(ACTIONS):]
    stable=added.all(0)
    strong=(added.mean(1)>=ref.mean(1)[:,None]-.01).all(0)
    extended={}
    names=list(MORE)
    for mask_name,mask in scopes.items():
        extended[mask_name]=dict(rows=int(mask.sum()),stable_any=int((stable.any(1)&mask).sum()),
              stable_comparable=int((stable[:,strong].any(1)&mask).sum()),
              only_occasional=int((~stable.any(1)&added.any((0,2))&mask).sum()),
              still_never=int((~added.any((0,2))&mask).sum()),
              per_model={n:int((stable[:,j]&mask).sum()) for j,n in enumerate(names)})
    result={}
    for key,rows in history.items():
        hit=np.stack([v for v,_ in rows])==y
        oracle=np.stack([v for _,v in rows])
        gain=hit.astype(int)-ref
        result[key]=dict(accuracy=hit.mean(1).tolist(),fixed=(gain>0).sum(1).tolist(),harmed=(gain<0).sum(1).tolist(),
            stable={n:int((hit.all(0)&m).sum()) for n,m in scopes.items()},
            oracle_any_member_per_split=(oracle&~ref).sum(1).tolist(),paired_ci_pp=interval(gain.mean(0),g))
    write(REPORT/"results.json",dict(candidates=methods,new_members=names,comparable_new=[n for n,m in zip(names,strong) if m],
                                     coverage=extended,policies=result,calibration_selections={k:dict(v) for k,v in freq.items()}))
    print(extended["unresolved"],result["all"],flush=True)


if __name__=="__main__":
    run()
