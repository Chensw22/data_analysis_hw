"""Test signed displacement between authenticated Mitra OOF checkpoints on all development rows."""
import json
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from audit_top5_references import row_hash
from broad_booking_policy_replay_n23 import interval
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical
from error_overlap_diagnostic import load_authenticated
from relational_error_corrector import sha, write

REPORT = ROOT / "reports/nov23_broad_mitra_displacement"
PHASE = ROOT / "reports/top5_adaptation_phase"
EQ = np.testing.assert_array_equal


def read(path):
    return json.loads(path.read_text())


def checked(path, expected, evidence):
    digest = sha(path)
    assert digest == expected, path
    evidence[str(path.relative_to(ROOT))] = digest
    return np.load(path, allow_pickle=False)


def grouped(seed, inputs, evidence):
    protocol = read(PHASE/"protocol.json")
    with checked(ART/f"top5_adaptation_phase/folds_{seed}.npz", protocol["fold_hashes"][str(seed)], evidence) as z:
        fold = z["fold_id"]
    with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
        for key, value in [("row",inputs["dev"]),("y",inputs["y"]),("fold",fold)]:
            EQ(z[key],value)
        base = z["base"]
    stages = {}
    for name, source in [("v71","refs/mitra"),("v76","candidates/adapt")]:
        results = read(PHASE/source/"summary.json")["results"]
        m = next(r for r in results if r["seed"] == seed)
        with checked(ROOT/m["oof_path"],m["oof_sha256"],evidence) as z:
            for key, value in [("dev_ids",inputs["dev"]),("fold_id",fold),("passenger_ids",inputs["passengers"]),
                               ("group_ids",inputs["groups"])]:
                EQ(z[key], value)
            stages[name] = z["prob"]
    for k in range(3):
        assert not set(inputs["groups"][fold==k]) & set(inputs["groups"][fold!=k])
    if seed == 20261112:
        digest = read(ROOT/"reports/group_episode_20261116/summary.json")["oof_sha256"]
        with checked(ART/"group_episode_20261116/group_20261112.npz",digest,evidence) as z:
            for key, value in [("dev_ids",inputs["dev"]),("fold_id",fold),("passenger_ids",inputs["passengers"]),
                               ("group_ids",inputs["groups"]),("reference_prob",stages["v71"])]:
                EQ(z[key],value)
            stages["v85"] = z["prob"]
        digest = read(ROOT/"reports/nov23_group_support3072/result.json")["sha256"]
        with checked(ART/"nov23_group_support3072/oof.npz",digest,evidence) as z:
            for key,value in [("row",inputs["dev"]),("y",inputs["y"]),("fold",fold)]:
                EQ(z[key],value)
            stages["v93"] = z["v93"]
    return fold,base,base>=.5,stages


def replay(inputs, splits, fold, probabilities, evidence):
    name = "v76_Mitra_lastblock100"
    meta = read(PHASE/"protocol.json")["screen_evidence"]["adapt"]
    assert meta["development_ids_sha256"] == row_hash(inputs["dev"])
    with checked(ROOT/meta["oof_path"], meta["oof_sha256"], evidence) as z:
        last = z["prob"]
        for k,(t,q) in enumerate(splits):
            path = ART/f"run_042/{name}_seed{20261112+k}_{row_hash(inputs['dev'][t])[:12]}.json"
            info = read(path)
            assert info["train_ids_sha256"] == row_hash(inputs["dev"][t])
            assert info["prediction_ids_sha256"] == row_hash(inputs["dev"][q])
            assert abs(z["fold_accuracy"][k]-((last[q]>=.5)==inputs["y"][q]).mean()) < 1e-12
            EQ(np.flatnonzero(fold==k), q)
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        for key,value in [("row",inputs["dev"]),("y",inputs["y"]),("fold",fold)]:
            EQ(z[key],value)
        return fold,z["v108"],z["v131"],dict(v71=probabilities["rawMitra"],v76=last,
                                              v81=probabilities["v81"],v93=probabilities["v93"])


def logits(p):
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    return logit(np.clip(p.astype(float), 1e-7,1-1e-7))


def upper(offset, delta, y, locked, ref):
    # For a common scalar the only possible decision changes occur at these roots.
    roots = -offset[delta!=0]/delta[delta!=0]
    knots = np.unique(np.r_[-3.,roots[(roots>=-3)&(roots<=3)],3.])
    grid = np.unique(np.r_[0.,knots,(knots[1:]+knots[:-1])/2])
    opt = (-1,np.inf,0.)
    for part in np.array_split(grid, max(1,len(grid)//160)):
        votes = offset+part[:,None]*delta>=0
        votes[:,locked] = ref[locked]
        correct = (votes==y).sum(1)
        i = np.lexsort((np.abs(part),-correct))[0]
        candidate = int(correct[i]), -float(abs(part[i])), float(part[i])
        opt = max(opt,candidate)
    return opt[2]


def shuffled(delta, offset, start, fold, seed):
    # Keep the marginal shift in joint score bins for both the starting model and strong baseline.
    rng = np.random.default_rng(seed)
    bins = np.searchsorted([-2,-1,0,1,2],offset)*6+np.searchsorted([-2,-1,0,1,2],start)
    out=delta.copy()
    for code in np.unique(bins+36*fold):
        q=np.flatnonzero(bins+36*fold==code)
        out[q]=delta[rng.permutation(q)]
    return out


def counts(gain, mask):
    return [int(np.count_nonzero(gain[mask]>0)),int(np.count_nonzero(gain[mask]<0))]


def run():
    row,y,masks = historical()
    inputs,splits,random_fold,probabilities,authenticated = load_authenticated()
    EQ(row, inputs["dev"])
    EQ(y, inputs["y"])
    group = inputs["groups"]
    evidence = {}
    plans = {str(s):grouped(s,inputs,evidence) for s in SEEDS}
    plans["v131"] = replay(inputs,splits,random_fold,probabilities,evidence)
    results, histories = {},{}
    for name,(fold,base,ref,stages) in plans.items():
        assert len(row)==6954 and len(np.unique(row))==6954
        offset = logits(base)
        locked = ref!=(base>=.5)
        partitions = []
        for k in range(3):
            q = np.flatnonzero(fold==k)
            for train,test in GroupKFold(3,shuffle=True,random_state=978).split(q,groups=group[q]):
                t, b = q[train],q[test]
                assert not set(group[t]) & set(group[b])
                partitions.append((t[~locked[t]], b))
        EQ(np.sort(np.concatenate([b for _,b in partitions])), np.arange(len(y)))
        methods = {}
        old = ref==y
        for first,last in combinations(stages,2):
            start,finish = logits(stages[first]),logits(stages[last])
            d = finish-start
            scalars = []
            learned = np.empty(len(y))
            for t,b in partitions:
                def loss(w):
                    a = offset[t]+w*d[t]
                    return (np.logaddexp(0,a)-y[t]*a).sum()+3*w*w
                w = minimize_scalar(loss, bounds=(-3,3),method="bounded").x
                learned[b] = w
                scalars.append(float(w))
            oracle = upper(offset,d,y,locked,ref)
            values = {"strong_half":offset+.5*d, "strong_full":offset+d,
                      "strong_learned":offset+learned*d, "strong_oracle":offset+oracle*d,
                      "strong_shuffled":offset+.5*shuffled(d,offset,start,fold,107),
                      "native_endpoint":finish,"native_beyond_half":finish+.5*d}
            change = f"{first}_{last}"
            details = {}
            for mode, score in values.items():
                vote = np.where(locked,ref,score>=0)
                hit = vote==y
                gain = hit.astype(int)-old
                details[mode] = dict(accuracy=float(hit.mean()),fixed_harmed=counts(gain,np.ones(len(y),bool)),
                                     net_by_fold=[int(gain[fold==k].sum()) for k in range(3)],
                                     cohort_fixed_harmed={n:counts(gain,m) for n,m in masks.items()})
                if mode=="strong_half":
                    details[mode]["group_bootstrap_delta_pp"] = interval(gain,group)
                if name!="v131":
                    histories.setdefault((change,mode),[]).append((hit,gain))
            # Individual retrospective reach, allowing a different multiplier per passenger.
            # Monotonicity means the two ends of each coefficient interval suffice.
            pos = ((offset+3*d>=0)==y)&~old&~locked
            neg = ((offset-3*d>=0)==y)&~old&~locked
            reach = {n:dict(positive=int((pos&m).sum()),either_sign=int(((pos|neg)&m).sum()))
                     for n,m in dict(all_rows=np.ones(len(y),bool),**masks).items()}
            methods[change]=dict(methods=details,learned_coefficients=scalars,oracle_coefficient=oracle,
                                 hindsight_individual_reach=reach)
        results[name]=dict(baseline_accuracy=float(old.mean()),locked=int(locked.sum()),directions=methods)
    stable={}
    for (change,mode), history in histories.items():
        if len(history)!=3:
            continue
        hit = np.stack([h for h,_ in history]).all(0)
        fixed = np.stack([g>0 for _,g in history]).all(0)
        harmed = np.stack([g<0 for _,g in history]).all(0)
        stable[mode]={n:dict(correct_all=int((hit&m).sum()),fixed_all=int((fixed&m).sum()),
                              harmed_all=int((harmed&m).sum())) for n,m in masks.items()}
        assert change=="v71_v76"
    write(REPORT/"result.json",dict(panels=results,stable_v71_v76=stable,
          evidence=evidence,historical_random_evidence={k:v for k,v in authenticated.items() if "Mitra" in k}))
    print({n:{c:[m["methods"][strategy]["fixed_harmed"] for strategy in
                 ("strong_half","strong_learned","strong_oracle")] for c,m in panel["directions"].items()}
                 for n,panel in results.items()})


if __name__ == "__main__":
    run()
