"""Repartition training bookings twice and assess dissent on exactly the same unseen passengers."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import panel
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_support_trajectories_n23 import LEVEL, fit_bank
from broad_tree_field_dividends_n23 import hard_cohorts
from exaone_validation_data import compact_features
from support_doubling_consistency_n23 import REPORT, get_bank


def alternatives(row, y, frame, groups, fold, seed):
    directory = ART/'nov23_replicated_quarter_dissent'
    directory.mkdir(exist_ok=True)
    outputs = []
    for repeat in [1, 2]:
        path = directory/f'{seed}_{repeat}.npz'
        if not path.exists():
            base_seed = 20261123 if seed == 'v131' else int(seed)
            bank, _ = fit_bank(frame, groups, y, fold, base_seed+repeat*1009, [(i,) for i in range(4)])
            np.savez_compressed(path, row=row, y=y, fold=fold, bank=bank)
        with np.load(path) as data:
            for key, expected in [('row', row), ('y', y), ('fold', fold)]:
                np.testing.assert_array_equal(data[key], expected)
            outputs.append(data['bank'])
    return np.stack(outputs)


def gate(x, active, y, ref, fold, groups):
    flip = np.zeros(len(y), bool)
    scored = np.zeros(len(y), bool)
    for f in np.unique(fold):
        ids = np.flatnonzero(active & (fold==f))
        for a,b in GroupKFold(3, shuffle=True, random_state=962).split(ids, groups=groups[ids]):
            t,q = ids[a], ids[b]
            assert not np.intersect1d(groups[t], groups[q]).size
            m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=400))
            m.fit(x[t], y[t]!=ref[t])
            flip[q] = m.predict(x[q])
            scored[q] = True
    np.testing.assert_array_equal(active, scored)
    return flip


def main():
    row, y, frame, groups, _ = load()
    _, _, hard, right = hard_cohorts(row, y)
    _, _, _, _, prior, _, _ = panel(row, y, frame)
    seen = np.logical_or.reduce(list(prior.values()))
    raw = pd.read_csv(ROOT/'data/raw/train.csv', usecols=lambda c:c not in ('Name','Transported'))
    personal = compact_features(raw.iloc[row])
    plans, observations, history = {}, [], {}
    for seed in [*map(str, SEEDS), 'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            fold, base, ref = z['fold'], z['base'], z['ref']
        original = get_bank(seed,row,y,groups,fold)[LEVEL==1]
        fresh = alternatives(row,y,personal,groups,fold,seed)
        bank = np.concatenate([original[None,:,:],fresh])
        count = ((bank>=.5)!=ref).sum(1)
        allowed = ref==(base>=.5)
        support = (count>0) & allowed
        eligible = support.any(0)
        aligned = logit(bank.clip(.001,.999))*np.where(ref,-1.,1.)
        ordered = np.sort(aligned.reshape(-1,len(y)),axis=0)
        risk = np.where(ref,1-base,base)
        baseline = np.c_[ref, logit(risk.clip(.001,.999))]
        features = np.column_stack([baseline, count.mean(0), support.sum(0), ordered[[0,3,5,8,11]].T])
        decisions = {'original': support[0], 'all_repartitions': support.all(0),
                     'risk_gate': gate(baseline,eligible,y,ref,fold,groups),
                     'repeat_gate': gate(features,eligible,y,ref,fold,groups)}
        plans[seed] = {}
        for name, flip in decisions.items():
            hit = (ref^flip)==y
            plans[seed][name] = dict(**paired(hit,ref==y), hard_fixed=int((hard & hit).sum()))
            if seed!='v131':
                history.setdefault(name,[]).append(hit)
        if seed!='v131':
            observations.append(support)
        print(seed, plans[seed], flush=True)
    # Axis order: original outer split, perturbation within that split, passenger.
    reproduced = np.stack(observations)
    original_stable = reproduced[:,0].all(0)
    nominees = {'old_eight': original_stable & hard, 'new_four':original_stable & hard & ~seen,
                'old_fifteen_harms':original_stable & right}
    assert [a.sum() for a in nominees.values()] == [8,4,15]
    np.testing.assert_array_equal(nominees['old_eight'], hard & np.all(history['original'], axis=0))
    np.testing.assert_array_equal(nominees['old_fifteen_harms'], right & ~np.any(history['original'], axis=0))
    retention = {k: dict(count=int(mask.sum()), fresh_repeats_out_of_six=np.bincount(
        reproduced[:,1:,mask].sum((0,1)), minlength=7).tolist()) for k,mask in nominees.items()}
    stable = {k:dict(hard_ever=int((hard & np.any(h,axis=0)).sum()), hard_stable=int((hard & np.all(h,axis=0)).sum()),
                     new_stable=int((hard & ~seen & np.all(h,axis=0)).sum()),
                     stable_harms=int((right & ~np.any(h,axis=0)).sum())) for k,h in history.items()}
    write(REPORT/'repeat.json',dict(plans=plans,retention=retention,stable=stable))
    print(retention,stable)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
