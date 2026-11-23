"""Assess recent public-score transfer with family holdouts and a paired comparison of two pending files."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_service_pairs_n23 import REPORT as PAIRS
from kaggle_workflow import sha256
from service_pair_portfolio_n23 import replay
from zero_donor_pair_overlay_n23 import additional

REPORT = ROOT/'reports/nov23_recent_family_transfer'
STATE = 'nov23_adaptive_manifest_splines/state_transfer.json'
THREE = 'nov23_dependence_anchors/three_corrections.json'
BASE = 'proposal_v131_party_consensus'
SECTIONS = [*map(str,SEEDS),'v131']
RELEASES = {
    'proposal_n23_incumbent_quarter_tabicl_budget': ('tabicl','nov23_bill_value/foundation_transfer.json','tab'),
    'proposal_n23_direct_basket_quarter': ('tabicl',STATE,'basket_quarter_direct'),
    'proposal_n23_joint_basket_quarter': ('tabicl',STATE,'basket_quarter_joint'),
    'proposal_n23_sleep_basket_joint_quarter': ('tabicl',STATE,'mean_quarter_joint'),
    'proposal_n23_partition_residual': ('partition',THREE,'partition'),
    'proposal_n23_posterior_spending_risk': ('spending_risk',THREE,'risk'),
}


def read(path):
    return json.loads(path.read_text())


def file_votes(name, ids):
    path = ART/name/'submission.csv'
    assert sha256(path) == read(ROOT/'reports'/name/'manifest.json')['sha256']
    df = pd.read_csv(path)
    np.testing.assert_array_equal(df.PassengerId, ids)
    assert df.Transported.dtype == bool
    return df.Transported.to_numpy(), sha256(path)


def confirmed(name, ids, official):
    vote, digest = file_votes(name,ids)
    receipt, = (ROOT/'reports'/name/'online_attempts').glob('*.json')
    r = read(receipt)
    assert r['file_sha256'] == digest and r['evaluation']['Status'] == 'COMPLETE'
    score = float(official[str(r['submission_ref'])])
    assert score == float(r['evaluation']['Public Score'])
    return score, vote, str(receipt.relative_to(ROOT))


def history(ids):
    official = read(ROOT/'reports/nov23_flip_capacity/primary_scores.json')['public_scores']
    baseline, base, _ = confirmed(BASE,ids,official)
    rows = []
    for name,(family,path,key) in RELEASES.items():
        score,vote,receipt = confirmed(name,ids,official)
        data = read(ROOT/'reports'/path)['plans']
        net = [data[s][key]['fixed']-data[s][key]['harmed'] for s in SECTIONS]
        rows.append(dict(name=name,family=family,local_net=net,group_mean=np.mean(net[:3])*100/6954,
                         group_min=np.min(net[:3])*100/6954,original=net[-1]*100/6954,
                         public_gain_pp=round((score-baseline)*100,4),test_changes=int((vote!=base).sum()),
                         receipt=receipt,local_source=path))
    return rows


def family_check(rows):
    families = sorted({r['family'] for r in rows})
    out = {}
    for target in families:
        query = [r for r in rows if r['family']==target]
        training = [r for r in rows if r['family']!=target]
        result = {}
        for metric in ('baseline','group_mean','group_min','original'):
            def predictor(r):
                return 0 if metric=='baseline' else r[metric]
            # Estimate intercept with equal training-family weight; use the fixed one-to-one
            # percentage-point scale rather than fit a slope to two training families.
            offsets = [np.mean([r['public_gain_pp']-predictor(r) for r in training if r['family']==family])
                       for family in families if family!=target]
            shift = float(np.mean(offsets))
            result[metric] = dict(training_family_offset_pp=shift,
                                  holdout_mae_pp=float(np.mean([abs(r['public_gain_pp']-(shift+predictor(r))) for r in query])))
        out[target] = result
    pairs = {}
    for metric in ('group_mean','group_min','original'):
        counts = {}
        for a,b in combinations(rows,2):
            families_key = '/'.join(sorted([a['family'],b['family']]))
            sign = np.sign(a[metric]-b[metric])*np.sign(a['public_gain_pp']-b['public_gain_pp'])
            key = 'tied' if sign==0 else 'concordant' if sign>0 else 'reversed'
            d = counts.setdefault(families_key,dict(tied=0,concordant=0,reversed=0))
            d[key]+=1
        pairs[metric]=counts
    return dict(held_out=out, pair_counts=pairs)


def head_to_head(ids):
    row,y,frame,groups,_ = load()
    weight = read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    grouped, plans = [], {}
    for s in SECTIONS:
        ref,masks = replay(frame,groups,row,y,s,weight)
        vote150 = ref ^ masks['pair']
        with np.load(ART/f'nov23_zero_spend_analogues/{s}.npz') as z:
            for k,v in [('ref',ref),('row',row),('y',y)]:
                np.testing.assert_array_equal(z[k],v)
            vote160 = vote150 | additional(z['base'],ref,vote150,z['matched'],row,.10)
        vote174 = ref ^ masks['pair_plus_corroborated']
        diff = (vote160==y).astype(int)-(vote174==y).astype(int)
        plans[s]=dict(better_160=int((diff>0).sum()),better_174=int((diff<0).sum()),
                      gain_160_over_174_pp=float(100*diff.mean()),gain_ci95_pp=interval(diff,groups))
        if s!='v131':
            grouped.append(diff)
    f160,_ = file_votes('proposal_n23_service_pair_zero_donor',ids)
    f174,_ = file_votes('proposal_n23_service_pair_corroborated',ids)
    return dict(plans=plans,group_averaged_gain_pp=float(100*np.mean(grouped)),
                group_averaged_ci95_pp=interval(np.mean(grouped,axis=0),groups),test_distance=int((f160!=f174).sum()))


def main():
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    models=history(ids)
    result=dict(models=models, family_check=family_check(models),compare=head_to_head(ids))
    write(REPORT/'results.json',result)
    print(result['compare'])
    print(result['family_check']['held_out'])


if __name__=='__main__':
    with threadpool_limits(1):
        main()
