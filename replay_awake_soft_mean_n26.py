"""Retest the marginal positive mean-score rule on changed booking splits and the actual competition roster."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from awake_paid_soft_unanimity_n26 import original_scores
from evaluate_nested_release_n26 import DECK,REVIEW,panel_predictions,stages
from export_expert_vote_panel_n26 import OUT as EXPERTS
from export_tabnet_attention_n26 import NAME as LATEST
from family_identity_transfer_n26 import panels_and_reference
from frozen_queue_shared_errors_n26 import paired, RELEASES
from held_expert_portfolios_n26 import NEW,sources,splits
from joint_candidate_coverage_n26 import branch_decisions
from price_implied_deck_n26 import FEES
from six_way_against_scored_nine_n26 import test_decisions
from solo_archive_recovery_n26 import archive,ticker
from tabnet_attention_transfer_n26 import collect,RESULT
from target_mix_expert_routing_n26 import ROOT


def historical(raw,y,g,active):
    views=sources()[-1]
    v=views['historical']
    ids,f=v['ids'],v['fold']
    _,_,_,rows,names,_,s,_,native,_=panels_and_reference()
    np.testing.assert_array_equal(rows,ids)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        nine,prior=z['nine'],z['v131']
    branches=branch_decisions(ids,f,s,names,native,prior,nine,nine,raw.CryoSleep.eq(False).to_numpy(),'historical',891)
    first=np.column_stack([branches[k][0] for k in ('nine','fields','awake','band','fields_awake','all_three')])
    p=panel_predictions(v,'historical')
    tags=np.array([n.split('/')[0] for n in names])
    _,legacy=archive('historical',ids,f,g,y,ROOT/'artifacts/nov24_remaining_cases/archive_scores.npz',probabilities=True)
    original=np.column_stack([*[s[:,tags==n].mean(1) for n in v['names'][:18]],legacy])
    np.testing.assert_array_equal(original>=.5,v['votes'])
    avg=np.column_stack([original,*[p[n] for n in NEW]]).mean(1)
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    scope=np.zeros(len(ids),bool)
    for key,_,_,_,q in splits(views,g):
        if not key.startswith('historical_') or key=='historical_to_new':
            continue
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],ids[q])
            risk=np.column_stack([z['identities_error'],z['identities_outcome']])
        anchor,review,fare,now=stages(v,p,q,enabled[ids[q]],risk)
        np.testing.assert_array_equal(anchor,first[q,-1])
        cold=~np.isin(g[ids[q]],g[ids[f!=f[q[0]]]])
        # The filtered-awake edits are a subset of awake edits in this historical comparison.
        scope[q]=cold&active[ids[q]]&(prior[q]>=0)&~first[q].any(1)&~review&~fare&~now
    return ticker(scope,y[ids],avg>=.5)


def run():
    raw,y,g,f,_,_=paired()
    active=raw.CryoSleep.eq(False).to_numpy()&raw[FEES].gt(0).any(axis=1).to_numpy()
    scores,_=original_scores(y,g,f)
    mean=scores.mean(1)
    runs,*_=collect()
    result={}
    for tag,stack in runs.items():
        q,common,now=(np.concatenate([v[i] for v in stack]) for i in range(3))
        prev=np.concatenate([v[3]['actual_and_scores'] for v in stack])
        target=common & ~now & active[q]
        r=ticker(target,y[q],mean[q]>=.5)
        extra=ticker(target & ~prev,y[q],mean[q]>=.5)
        result[tag]=dict(original=r,beyond_tabnet=extra)
    assert (result['full']['original']['fixed'],result['full']['original']['harmed'])==(19,15)
    result['historical_cold_before_tabnet']=historical(raw,y,g,active)
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    with np.load(EXPERTS/'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        mean=z['scores'].mean(1)
    versions=np.column_stack([test_decisions(n,test.PassengerId)[0] for n in RELEASES.values()])
    latest,_=test_decisions(LATEST,test.PassengerId)
    select=test.CryoSleep.eq(False).to_numpy()&test[FEES].gt(0).any(axis=1).to_numpy()&~versions.any(1)&(mean>=.5)
    result['test']=dict(ten_version_edits=int(select.sum()),additional_after_queued_tabnet=int((select&~latest).sum()))
    print(result,flush=True)
    (RESULT.parent/'replay_awake_soft_mean.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
