"""Replay the confirmed-fields or dispute-band edit on untouched, non-spending passengers in shared bookings."""
import json
import numpy as np
import pandas as pd
from export_consensus_review_n26 import ART,ROOT,dig,write
from frozen_queue_shared_errors_n26 import RELEASES
from kaggle_workflow import validate_submission
from rebase_fare_six_n26 import NAME as BASE
from six_way_against_scored_nine_n26 import test_decisions
from price_implied_deck_n26 import FEES

NAME='proposal_n27_unpaid_group_transfer'


def predict(parent,nine,fields,band,eligible):
    base=parent>=.5
    f=(fields>=.5)!=nine
    b=(band>=.5)!=nine
    active=eligible&(base==nine)&(f|b)
    suggestions=(f*fields+b*band)/(f.astype(float)+b).clip(1)
    p=np.where(active,suggestions,parent)
    np.testing.assert_array_equal(p>=.5,base^active)
    return p


def run():
    raw=pd.read_csv(ROOT/'data/raw/test.csv')
    ids=raw.PassengerId
    rows=len(pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']))
    arr={}
    parents={}
    for k,name in [('parent',BASE),('nine',RELEASES['nine']),('fields',RELEASES['fields']),('band',RELEASES['band'])]:
        hard,digest=test_decisions(name,ids)
        parents[name]=digest
        if k=='nine':
            arr[k]=hard
            continue
        stored=pd.read_csv(ART/name/'test_probabilities.csv',float_precision='round_trip')
        np.testing.assert_array_equal(stored.PassengerId,ids)
        np.testing.assert_array_equal(stored.probability,stored.reload_probability)
        arr[k]=stored.probability.to_numpy()
        np.testing.assert_array_equal(arr[k]>=.5,hard)
    eligible=ids.str.split('_').str[0].duplicated(keep=False).to_numpy() & ~raw[FEES].gt(0).any(axis=1).to_numpy()
    arr['eligible']=eligible
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    p=predict(**arr)
    np.savez_compressed(model,passenger=ids.to_numpy(str),**arr)
    with np.load(model) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        replay=predict(**{k:z[k] for k in arr})
    np.testing.assert_array_equal(p,replay)
    with np.load(ART/'nov27_independent_archive_transfer/test.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['111_multi_unpaid_fields_or_band'],p>=.5)
    pd.DataFrame({'PassengerId':ids,'Transported':p>=.5}).to_csv(dest/'submission.csv',index=False)
    pd.DataFrame({'PassengerId':ids,'probability':p,'reload_probability':replay}).to_csv(dest/'test_probabilities.csv',index=False)
    checked=validate_submission(dest/'submission.csv',ROOT/'data/raw/test.csv')
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        doc=json.loads(receipt.read_text())
        assert doc['file_sha256']!=checked['sha256'] or receipt.parents[1].name==NAME
    edit=(p>=.5)!=(arr['parent']>=.5)
    version='n27_unpaid_group_transfer'
    manifest=dict(**checked,version=version,training_rows=rows,parents=parents,model_sha256=dig(model),
                  reload_difference=0,changed=int(edit.sum()),changed_orders=int(ids[edit].str[:4].nunique()),source_file='export_unpaid_group_transfer_n27.py')
    assert manifest['changed']==26 and manifest['changed_orders']==24
    sel=dict(version=version,rule='Keep the scored 111 release. Where it agrees with nine, accept either archived fields or band opposition on a multi-passenger booking with no observed positive facility payment.',
             evidence='reports/nov23_matched_learning_curve/independent_archive_transfer_n27.json',
             validation='Exploratory after comparison of 20 policies. No new-order validation booking was changed by this rule.')
    for folder in (dest,ROOT/'reports'/NAME):
        folder.mkdir(exist_ok=True)
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',sel)
    print(manifest)


if __name__=='__main__':run()
