"""Apply the double-confirmed TabNet countervote to previously unanimous passengers."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from attentive_tabnet_n26 import OUT as NETWORK
from booking_mix_confidence_transfer_n26 import partition
from evaluate_nested_release_n26 import protocols,sources
from export_consensus_review_n26 import ART,dig,write
from frozen_queue_shared_errors_n26 import RELEASES
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_reliability_n26 import network_outputs
from tabnet_attention_transfer_n26 import RESULT,decide,historical_outputs
from target_mix_expert_routing_n26 import ROOT,batches
from test_release_risk_fields_n26 import CACHE,REPORT

NAME='proposal_n26_tabnet_attention'
PARENT=RELEASES['six']
OUT=ART/NAME


def combine(parent, counter, approved, common):
    return np.where(approved&common,counter,parent)


def export():
    OUT.mkdir(exist_ok=True)
    receipts=json.loads((RESULT.parent/'tabnet_history_test.json').read_text())
    assert dig(CACHE)==json.loads(REPORT.read_text())['sha256']
    assert dig(NETWORK/'test.npz')==receipts['test']['sha256']
    for k,item in receipts['history'].items():
        assert dig(NETWORK/'history'/f'fold{k}.npz')==item['sha256']
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    ids=test.PassengerId.to_numpy(str)
    groups=test.PassengerId.str.split('_').str[0].to_numpy()
    with np.load(CACHE) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        fields,now,common=z['fields'],z['parent'],z['common']
    with np.load(NETWORK/'test.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        pt,at=network_outputs(z)
    raw,y,g,_=data()
    panels=sources()[-1]
    assert not np.isin(g,groups).any()
    ph,ah=historical_outputs(raw,y,g,panels['historical'])
    source=partition(groups,'source:',3)
    shard=partition(groups,'query:',5)
    seen=np.zeros(len(test),int)
    approved=np.zeros(len(test),bool)
    for details,b in zip(protocols(panels,g,False),batches(False),strict=True):
        key,src,_,it,_=details
        if b.tag!='bookingmix':
            continue
        _,fold,k=key.split('_')
        t=b.train_ids
        np.testing.assert_array_equal(t,src['ids'][it])
        mask=(source==int(fold))&(shard==int(k))
        seen[mask]+=1
        approved[mask]=decide(b.train_fields,fields[mask],b.train_cell%2!=0,now[mask],b.train_error,
                              ph[t],pt[mask],ah[t],at[mask])['actual_and_scores']
    assert (seen==1).all()
    parent,checksum=test_decisions(PARENT,ids)
    np.testing.assert_array_equal(parent,now)
    ref=pd.read_csv(ART/PARENT/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(ref.PassengerId,ids)
    base=ref.probability.to_numpy()
    np.testing.assert_array_equal(base>=.5,now)
    counter=pt.mean(1)
    output=combine(base,counter,approved,common)
    edit=(output>=.5)!=now
    assert np.array_equal(edit,approved&common) and edit.any()
    model=OUT/'predictions.npz'
    np.savez_compressed(model,passenger=ids,parent=base,counter=counter,approved=approved,common=common)
    with np.load(model) as z:
        replay=combine(z['parent'],z['counter'],z['approved'],z['common'])
    np.testing.assert_array_equal(output,replay)
    file=OUT/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids,Transported=output>=.5)).to_csv(file,index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=output,reload_probability=replay)).to_csv(OUT/'test_probabilities.csv',index=False)
    check=validate_submission(file,ROOT/'data/raw/test.csv')
    for rec in (ROOT/'reports').glob('*/online_attempts/*.json'):
        assert json.loads(rec.read_text())['file_sha256']!=check['sha256']
    for release in RELEASES.values():
        prev,sha=test_decisions(release,ids)
        assert (prev!=(output>=.5)).any() and sha!=check['sha256']
    nine,_=test_decisions(RELEASES['nine'],ids)
    info=dict(**check,version=NAME.removeprefix('proposal_'),training_rows=len(y),parents={PARENT:checksum},
              model_sha256=dig(model),reload_difference=0,parent_changes=int(edit.sum()),previous_best_changes=int(((output>=.5)!=nine).sum()),
              previously_unanimous=int(common.sum()),test_tabnet_sha256=receipts['test']['sha256'],test_fields_sha256=dig(CACHE))
    selection=dict(version=info['version'],rule='Keep the parent unless all ten previous releases agree, the TabNet mean dissents, and both the score-only and score-plus-attention error models endorse the change. Assign held historical calibrators by booking as in bookingmix.',
                   evidence=str(RESULT.relative_to(ROOT)))
    for directory in (OUT,ROOT/'reports'/NAME):
        write(directory/'manifest.json',info)
        write(directory/'selection.json',selection)
    print(info,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        export()
