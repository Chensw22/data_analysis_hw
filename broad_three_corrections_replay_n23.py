"""Test equal-weight combinations of three existing correction recipes with aligned rows."""
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_dependence_anchors_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from export_personal_bill_mix_n23 import NAME as PERSONAL
from export_partition_residual_n23 import NAME as PARTITION
from export_posterior_spend_risk_n23 import NAME as RISK, profile, probabilities

NAMES = (PERSONAL,PARTITION,RISK)


def scores(frame,g,row,y,seed):
    with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as data:
        np.testing.assert_array_equal(row,data['row'])
        np.testing.assert_array_equal(y,data['y'])
        fold,base,ref = data['fold'],data['base'],data['ref']
        locked = ref!=(base>=.5)
        partition = np.where(locked,ref,data['update'])
    with np.load(ART/f'nov23_two_source_manifest/{seed}.npz') as data:
        for key,actual in [('fold',fold),('row',row),('ref',ref),('base',base),('y',y)]:
            np.testing.assert_array_equal(data[key],actual)
        personal = np.where(locked,ref,.75*base+.25*data['direct'])
    table = profile(frame,base,ref).assign(error=ref!=y)
    risk = np.full(len(y),np.nan)
    for f in np.unique(fold):
        ids = np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=911).split(ids,groups=g[ids]):
            t,q = ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            risk[q] = probabilities(table.iloc[t],table.iloc[q].drop(columns='error'))
    matrix = np.column_stack([personal,partition,risk])
    assert np.isfinite(matrix).all() and ((matrix>=0)&(matrix<=1)).all()
    return matrix,ref


def main():
    row,y,frame,g,_ = load()
    wrong,right,hard = cohorts(row,y)
    plans, history = {},{}
    records = {}
    for name in NAMES:
        receipts = list((ROOT/'reports'/name/'online_attempts').glob('*.json'))
        records[name] = [dict(score=json.loads(p.read_text())['evaluation']['Public Score'],
                              receipt=str(p.relative_to(ROOT))) for p in receipts]
    for i,seed in enumerate([*map(str,SEEDS),'v131']):
        matrix,ref = scores(frame,g,row,y,seed)
        hit = {k:(p>=.5)==y for k,p in zip(('personal','partition','risk'),matrix.T)}
        hit['mean_probability'] = (matrix.mean(1)>=.5)==y
        hit['majority'] = ((matrix>=.5).sum(1)>=2)==y
        hit['reference'] = ref==y
        metrics = {k:paired(v,ref==y) for k,v in hit.items()}
        for k,expected in [('personal',[3,2,9,17]),('partition',[8,13,6,1]),('risk',[11,37,8,3])]:
            m=metrics[k]
            assert m['fixed']-m['harmed']==expected[i]
        plans[seed] = metrics
        if seed!='v131':
            for k,v in hit.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    baseline = np.stack(history['reference'])
    stable = {}
    for k,h in history.items():
        h = np.stack(h)
        stable[k] = dict(hard_stable=int((hard & h.all(0)).sum()),hard_ever=int((hard & h.any(0)).sum()),
                         common_stable=int((wrong & h.all(0)).sum()),stable_harms=int((right & ~h.any(0)).sum()),
                         gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'three_corrections.json',dict(plans=plans,stable=stable,online_parents=records))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
