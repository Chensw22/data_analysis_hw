"""Repeat population routing on fresh booking splits and a baseline-matched placebo."""
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import predictions
from broad_bill_exception_n23 import load, strata
from broad_booking_policy_n23 import ACTIONS
from broad_booking_policy_replay_n23 import interval
from broad_expert_copula_n23 import REPORT
from broad_interaction_error_transfer_n23 import fetch
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_release_miss_coverage_n23 import summarize
from broad_specialist_tree_average_n23 import average, routing_partitions
from released_correction_audit_n23 import development


def replay(scores, g, targets, query_splits):
    vote = .25+.5*(scores>=.5)
    output = {}
    for shuffle in (False, True):
        partitions = routing_partitions(scores, shuffle)
        for seed in query_splits:
            out = np.empty(len(targets),float)
            for t,q in GroupKFold(5,shuffle=True,random_state=seed).split(vote,groups=g):
                assert not np.intersect1d(g[t],g[q]).size
                out[q] = np.mean([average(tree,vote,t,q,targets[t]) for tree in partitions],axis=0)
            output[shuffle,seed] = out
    return output


def main():
    rows, y, frame, groups, _ = load()
    targets, releases, masks = development()
    np.testing.assert_array_equal(targets,y)
    missed=np.logical_and.reduce([p!=y for p in releases.values()])
    bill=strata(frame)
    core=missed&(((bill==2)&y)|((bill==0)&~y))
    scopes=dict(ever=masks["ever"],repeat=masks["repeat"],common=missed,core=core,
                repeat_outside=masks["repeat"]&~core)
    history,refs={},[]
    for s in SEEDS:
        base, p, fold=predictions(rows,y,s,ACTIONS)
        gen=fetch("class_transport",["independent","tree_mixture"],rows,y,fold,base,s)
        latent=fetch("hidden_context",["raw","shared"],rows,y,fold,base,s)
        ref=(base>=.5)==y
        refs.append(ref)
        maps={}
        for title,values in (("original",np.c_[base,p]),("augmented",np.column_stack([base,p,*gen,*latent]))):
            for f in np.unique(fold):
                ix=np.flatnonzero(fold==f)
                out=replay(values[ix],groups[ix],y[ix],[481,179])
                for (sham,seed),pred in out.items():
                    key=f"{title}/{'sham' if sham else 'real'}/{seed}"
                    maps.setdefault(key,np.full(len(y),np.nan))[ix]=pred
        with np.load(ART/f"nov23_specialist_tree_average/{s}.npz") as z:
            np.testing.assert_allclose(z["vote/average"],maps["original/real/481"],rtol=0,atol=1.e-12)
        for k, prob in maps.items():
            assert np.isfinite(prob).all()
            choice=np.where(np.isclose(prob,.5,atol=1e-12,rtol=0),base>=.5,prob>.5)
            history.setdefault(k,[]).append(choice==y)
        print(s,[(k,int(history[k][-1].sum()-ref.sum())) for k in maps if k.endswith('/179')],flush=True)
    refs=np.stack(refs)
    stats={k:summarize(np.stack(v),refs,scopes) for k,v in history.items()}
    frequency = (~refs).sum(axis=0)
    breakdown = {}
    for k, values in history.items():
        hits = np.stack(values)
        groups_of_errors = {}
        for n in range(4):
            selected = frequency == n
            before, after = refs[:, selected], hits[:, selected]
            groups_of_errors[str(n)] = dict(passengers=int(selected.sum()),
                fixed=(after & ~before).sum(1).tolist(), harmed=(before & ~after).sum(1).tolist(),
                correct_in_zero_to_three=np.bincount(after.sum(0), minlength=4).tolist())
        breakdown[k] = groups_of_errors
    comparisons={}
    for seed in [481,179]:
        for a,b in (("original/real","original/sham"),("augmented/real","original/real"),
                    ("augmented/real","augmented/sham"),("original/real","baseline")):
            diff=np.array(history[f"{a}/{seed}"],float)-(refs if b=="baseline" else np.array(history[f"{b}/{seed}"]))
            comparisons[f"{a}-vs-{b}/{seed}"]=dict(mean_pp=float(diff.mean()*100),
                                                    interval_pp=interval(diff.mean(0),groups))
    write(REPORT/"routing_replay.json",dict(policies=stats,paired_comparisons=comparisons,by_baseline_errors=breakdown))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
