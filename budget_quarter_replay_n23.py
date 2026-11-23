"""Replay the previously fixed 25% budget-only expert on the incumbent and on larger fits."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from exaone_validation_data import compact_features
from matched_booking_learning_curve_n23 import portions
from spend_simplex_likelihood_n23 import design, learn

REPORT = ROOT/'reports/nov23_budget_quarter_replay'


def score(name, alt, prior, ref, y, groups, fold, report):
    p = .75*prior + .25*alt
    eligible = (prior>=.5) == ref
    vote = np.where(eligible, p>=.5, ref)
    gain = (vote==y).astype(int) - (ref==y).astype(int)
    report[name] = dict(fixed=int((gain==1).sum()), harmed=int((gain==-1).sum()),
                        accuracy=float((vote==y).mean()), delta_pp=float(100*gain.mean()),
                        interval_pp=interval(gain,groups),
                        folds=[int(gain[fold==i].sum()) for i in np.unique(fold)],
                        overrides_preserved=int((~eligible).sum()))
    print(name,report[name],flush=True)
    return vote


def run():
    inventory = pd.read_csv(ART/'nov23_global_error_inventory.csv')
    row = inventory.row.to_numpy()
    y = inventory.target.to_numpy(bool)
    raw = pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda s:s not in ['Name','Transported']).iloc[row]
    groups = raw.PassengerId.str.split('_').str[0].to_numpy()
    x = design(compact_features(raw).reset_index(drop=True))[0]
    out = ART/'nov23_budget_quarter_replay'
    out.mkdir(exist_ok=True)
    report = {}
    for seed in SEEDS:
        with np.load(ART/f'nov23_joint_risk_surface/{seed}.npz') as j, np.load(ART/f'nov23_simplex_likelihood/{seed}.npz') as s:
            for a in [j,s]:
                np.testing.assert_array_equal(a['row'],row)
                np.testing.assert_array_equal(a['y'],y)
            np.testing.assert_array_equal(j['fold'],s['fold'])
            vote=score(str(seed),s['budget_prior'],j['base'],j['base']>=.5,y,groups,j['fold'],report)
            np.testing.assert_array_equal(vote,j['fixed_quarter']>=.5)
    with np.load(ART/'nov23_broad_partition_evidence/v131.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        fold,base,ref=z['fold'],z['base'],z['ref']
    orig=np.full(len(y),np.nan)
    for f in np.unique(fold):
        q=np.flatnonzero(fold==f)
        t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
        orig[q]=learn(x,y,t,q)
    vote=score('original_v131',orig,base,ref,y,groups,fold,report)
    np.savez_compressed(out/'original.npz',row=row,y=y,fold=fold,base=base,ref=ref,alternative=orig,vote=vote)
    with np.load(ART/'nov23_near_full_distillation/predictions.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        fold,ref,sizes=z['fold'],z['reference'],z['sizes']
    probs=np.full(ref.shape,np.nan)
    for f,small,large,q in portions(groups):
        np.testing.assert_array_equal(np.flatnonzero(fold==f),q)
        np.testing.assert_array_equal(sizes[f],[len(small),len(large),len(q)])
        for j,t in enumerate([small,large]):
            probs[j,q]=learn(x,y,t,q)
    assert np.isfinite(probs).all()
    for j,label in enumerate(['small','large']):
        score('nested8_'+label,probs[j],ref[j],ref[j]>=.5,y,groups,fold,report)
    np.savez_compressed(out/'matched.npz',row=row,y=y,fold=fold,base=ref,alt=probs)
    write(REPORT/'result.json',report)


if __name__=='__main__':
    with threadpool_limits(2):run()
