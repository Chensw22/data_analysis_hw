"""Export the frozen two-half disagreement veto for an otherwise unused submission slot."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from check_disagreement_fallback_n26 import FIRST,apply,fit,historical_training
from export_consensus_review_n26 import ART,dig,write
from forecast_frozen_queue_n26 import NAMES
from full_roster_booking_validation_n23 import ROOT,data
from held_expert_portfolios_n26 import sources
from kaggle_workflow import validate_submission
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import RESULT

NAME='proposal_n26_historical_veto'
USED=[*FIRST,'review','fare','six']


def replay(model,ids):
    votes={}
    for k in [*USED,'tabnet']:
        votes[k],fingerprint=test_decisions(NAMES[k],ids)
        assert fingerprint==model['parents'][NAMES[k]]
    rules={(side,half):np.asarray(model['coalitions'][str(int(side))][half],dtype=np.int32)
           for side in (False,True) for half in (0,1)}
    veto=apply(np.column_stack([votes[k] for k in USED]),rules)
    assert not (veto&(votes['six']!=votes['tabnet'])).any()
    return votes['tabnet']^veto


def export():
    raw,y,g,_=data()
    old,truth,orders,valid=historical_training(y,g,sources()[-1]['historical'])
    rules=fit(old,truth,orders,valid)
    rows=valid&(old!=old[:,[-1]]).any(1)
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    assert not np.isin(g,ids.str.split('_').str[0]).any()
    parents={NAMES[n]:test_decisions(NAMES[n],ids)[1] for n in [*USED,'tabnet']}
    composition={'parents':parents,'columns':USED,'coalitions':{str(int(side)):[rules[side,h].tolist() for h in (0,1)]
                 for side in (False,True)}}
    out=ART/NAME
    out.mkdir(exist_ok=True)
    model=out/'composition.json'
    write(model,composition)
    initial=replay(composition,ids)
    reloaded=replay(json.loads(model.read_text()),ids)
    np.testing.assert_array_equal(initial,reloaded)
    csv=out/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=initial)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=initial.astype(float),reload_probability=reloaded.astype(float))).to_csv(out/'test_probabilities.csv',index=False)
    checked=validate_submission(csv,ROOT/'data/raw/test.csv')
    current={k:test_decisions(v,ids)[0] for k,v in NAMES.items()}
    distances={k:int((v!=initial).sum()) for k,v in current.items()}
    assert distances['tabnet']==26 and min(distances.values())>0
    for rec in (ROOT/'reports').glob('*/online_attempts/*.json'):
        assert json.loads(rec.read_text())['file_sha256']!=checked['sha256']
    selection=dict(version=NAME.removeprefix('proposal_'),
        rule='Keep the frozen TabNet parent outside historical-release disagreement. Within disagreement, reverse the six-way decision only when both separately fitted historical half-order policies endorse it for the same prediction direction.',
        use='One conditional replacement for a skipped confirmed-union submission, after the terminal TabNet score and only with a remaining primary slot in the same UTC scoring day.',
        evidence='reports/nov23_matched_learning_curve/check_disagreement_fallback.json')
    manifest=dict(**checked,version=selection['version'],parents=parents,training_rows=int(rows.sum()),
                  training_bookings=len(np.unique(orders[rows])),model_sha256=dig(model),reload_difference=0,
                  hard_probabilities=True,distances=distances)
    for directory in (out,ROOT/'reports'/NAME):
        write(directory/'selection.json',selection)
        write(directory/'manifest.json',manifest)
    print(manifest,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):export()
