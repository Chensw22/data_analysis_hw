"""Learn when to accept a dissenting booking-bill or passenger-neighborhood vote."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from broad_booking_canonical_n23 import load_roster
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_link_authenticity_n23 import REPORT
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, SEEDS, write
from broad_unanimous_dissent_value_n23 import SOURCES


def bill_roles(row, groups):
    frame, full_groups, size, _ = load_roster()
    np.testing.assert_array_equal(full_groups[row], groups)
    amount = frame[BILLS].fillna(0).to_numpy()
    total = pd.DataFrame(amount).groupby(full_groups).transform('sum').to_numpy()
    peer = total - amount
    assert (peer >= 0).all()
    observed = frame[BILLS].notna().sum(axis=1).to_numpy() / len(BILLS)
    features = np.column_stack([np.log(size), observed, np.log1p(amount)/7,
                                np.log1p(peer)/7, amount.sum(1)/np.maximum(total.sum(1), 1)])
    return features[row], size[row] > 1


def opinions(seed, row, y, many):
    with np.load(ART/f'nov23_within_booking_bill_matching/{seed}.npz') as data:
        np.testing.assert_array_equal(row, data['row'])
        np.testing.assert_array_equal(y, data['y'])
        fold, base, ref = data['fold'], data['base'], data['ref']
        billing = np.column_stack([np.where(many, data[k], base) for k in ('owner','symmetrized')])
    neighbors = []
    for folder, key in SOURCES.values():
        with np.load(ART/f'nov23_{folder}/{seed}.npz') as data:
            np.testing.assert_array_equal(row, data['row'])
            np.testing.assert_array_equal(fold, data['fold'])
            np.testing.assert_array_equal(y, data['y'])
            neighbors.append(data[key])
    return fold, base, ref, billing, np.column_stack(neighbors)


def decide(base, ref, prob, roles, fold, groups, y):
    # Only fit on disagreements. Orient every expert's score towards contradicting
    # the incumbent, so that either direction of correction can share evidence.
    eligible = ((prob >= .5) != ref[:,None]).any(1) & (ref == (base >= .5))
    sign = np.where(ref, -1., 1.)
    strength = sign[:,None] * logit(np.clip(prob, .003, .997))
    confidence = abs(logit(np.clip(base, .003, .997)))
    anchor = np.column_stack([confidence, sign, confidence*sign])
    votes = np.column_stack([anchor, strength, sign[:,None]*strength])
    designs = {'anchor':anchor, 'experts':votes,
               'roles':np.column_stack([votes, roles, strength.max(1)[:,None]*roles])}
    flip = {k:np.zeros(len(y),bool) for k in designs}
    error = y != ref
    for outer in np.unique(fold):
        all_ids = np.flatnonzero(fold == outer)
        for a,b in GroupKFold(3,shuffle=True,random_state=502).split(all_ids,groups=groups[all_ids]):
            t,q = all_ids[a], all_ids[b]
            assert not np.intersect1d(groups[t],groups[q]).size
            t,q = t[eligible[t]], q[eligible[q]]
            if not len(q):
                continue
            for key,x in designs.items():
                fit = LogisticRegression(C=.2, max_iter=500).fit(x[t],error[t])
                flip[key][q] = fit.predict(x[q])
    return {k:ref ^ change for k,change in flip.items()},eligible


def main():
    row,y,_,groups,_ = load()
    wrong,right,hard = cohorts(row,y)
    roles,many = bill_roles(row,groups)
    plans, history = {}, {}
    for seed in [*map(str,SEEDS),'v131']:
        fold,base,ref,bill,near = opinions(seed,row,y,many)
        methods = {'reference':ref}
        eligible_counts = {}
        for name,prob in [('bill',bill),('neighbors',near),('combined',np.column_stack([near,bill]))]:
            pred,eligible = decide(base,ref,prob,roles,fold,groups,y)
            eligible_counts[name] = int(eligible.sum())
            methods.update({name+'_'+k:v for k,v in pred.items()})
        changes = {k:paired(v==y,ref==y) for k,v in methods.items()}
        plans[seed] = dict(changes=changes, eligible_counts=eligible_counts)
        if seed!='v131':
            for k,v in methods.items():
                history.setdefault(k,[]).append(v==y)
        print(seed,{k:m['fixed']-m['harmed'] for k,m in changes.items()},flush=True)

    baseline = np.stack(history['reference'])
    stable = {}
    for key, parts in history.items():
        hit = np.stack(parts)
        stable[key] = dict(hard_stable=int((hard & hit.all(0)).sum()),hard_ever=int((hard & hit.any(0)).sum()),
                           common_stable=int((wrong & hit.all(0)).sum()),stable_harms=int((right & ~hit.any(0)).sum()),
                           gain95_pp=interval((hit.astype(float)-baseline).mean(0),groups))
    write(REPORT/'decision.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    main()
