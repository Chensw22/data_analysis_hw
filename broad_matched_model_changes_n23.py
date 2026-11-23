"""Combine whole-population sources as control forecasts and matched changes to each forecast."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

FAMILIES = {
    "adaptive_manifest_splines": ("additive", "replicated_pair"),
    "independent_triggers": ("logit", "transport_trigger_matched"),
    "multiscale_count_training": ("ordinary", "multiscale"),
    "field_robustness": ("clean", "worst"),
    "local_manifest_ranks": ("original", "real"),
    "residual_kernel_alignment": ("uniform", "replicated"),
}


def read_bank(row, y, fold, seed):
    bank = []
    for folder, fields in FAMILIES.items():
        with np.load(ART / f"nov23_{folder}/{seed}.npz") as z:
            for key, data in [('row',row), ('y',y), ('fold',fold)]:
                np.testing.assert_array_equal(z[key],data)
            raw = [z[f][:,0] if z[f].ndim == 2 else z[f] for f in fields]
            bank.append(np.column_stack(raw))
    return np.stack(bank, axis=1)


def differences(control, treated, anchor):
    # The signed change and its agreement with changes from independent model families.
    delta = treated - control
    other = (delta.sum(axis=1, keepdims=True) - delta) / (delta.shape[1] - 1)
    shared = np.sign(delta)*np.minimum(abs(delta),abs(other))*(delta*other > 0)
    consensus = np.median(control, axis=1)
    return np.column_stack([delta, shared, delta * np.tanh((consensus-anchor)**2)[:,None]])


def readout(anchor, features, y, train, query):
    xt, xq = features[train], features[query]
    def objective(w):
        margin = anchor[train] + w[0] + xt @ w[1:]
        grad = expit(margin) - y[train]
        return (np.logaddexp(0.,margin).sum() - y[train]@margin + 18*(w@w),
                np.r_[grad.sum(),xt.T@grad] + 36*w)
    result = minimize(objective,np.zeros(1+features.shape[1]),method='L-BFGS-B',jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    return anchor[query] + result.x[0] + xq @ result.x[1:] >= 0


def main():
    row,y,_,groups,_ = load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        wrong = (z['votes']!=y[None,:,None]).all((0,2))
        right = (z['votes']==y[None,:,None]).all((0,2))
    plans, bank, useful = {}, {}, []
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,opinions,_ = read_opinions(seed,row,y)
        scores = read_bank(row,y,fold,seed)
        margin = logit(scores.clip(.003,.997))
        anchor = logit(opinions[:,0].clip(.003,.997))
        control,treated = margin[:,:,0],margin[:,:,1]
        base_design = control - anchor[:,None]
        changes = differences(control,treated,anchor)
        designs = dict(control=base_design, structured=treated-anchor[:,None],
                       separate=np.c_[base_design,treated-control], corroborated=np.c_[base_design,changes])
        output = {name:ref.copy() for name in designs}
        for f in np.unique(fold):
            use = np.flatnonzero((fold==f)&(ref==(anchor>=0)))
            for a,b in GroupKFold(3,shuffle=True,random_state=925).split(use,groups=groups[use]):
                t,q = use[a],use[b]
                assert not np.intersect1d(groups[t],groups[q]).size
                for name,design in designs.items():
                    output[name][q] = readout(anchor,design,y,t,q)
        hits = {'reference':ref==y,**{k:v==y for k,v in output.items()}}
        metrics = {k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed] = metrics
        if seed!='v131':
            useful.append((scores >= .5)==y[:,None,None])
            for k,v in hits.items():
                bank.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    a = np.stack(useful)
    ref = np.stack(bank['reference'])
    stable = {k:dict(correct_common_wrong=int((np.all(v,axis=0)&wrong).sum()),
                     missed_common_right=int((~np.any(v,axis=0)&right).sum()),
                     gain95_pp=interval((np.stack(v).astype(float)-ref).mean(0),groups))
              for k,v in bank.items()}
    control, structured = a.all(0)[:,:,0].any(1), a.all(0)[:,:,1].any(1)
    harms = (~a).all(0).any(axis=1)
    coverage = dict(control_stable=int((wrong&control).sum()),structured_stable=int((wrong&structured).sum()),
                    stable_in_either=int((wrong&(control|structured)).sum()),
                    structured_only=int((wrong&structured&~control).sum()),
                    harmed_in_either=int((right&harms.any(1)).sum()),structured_only_harm=int((right&harms[:,1]&~harms[:,0]).sum()))
    write(ROOT/'reports/nov23_adaptive_manifest_splines/combined.json',dict(families=FAMILIES,plans=plans,stable=stable,coverage=coverage))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
