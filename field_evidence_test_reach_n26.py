"""Count prospective changes from the two field-required policies on the untouched competition roster."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from export_tabnet_attention_n26 import NAME as LATEST
from frozen_queue_shared_errors_n26 import RELEASES
from full_roster_booking_validation_n23 import ROOT,data
from omit_surprising_passenger_field_n26 import CACHE,compact_features,fit_split
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import RESULT


def run():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist()).loc[:,raw.columns]
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(g,tg).any()
    people=pd.concat([raw,test],ignore_index=True)
    groups=np.r_[g,tg]
    frame=compact_features(people)
    frame['OrderSize']=people.groupby(groups).PassengerId.transform('size')
    t,q=np.arange(len(y)),np.arange(len(y),len(people))
    p,choice,sham=fit_split(frame,y,groups,t,q,0)
    ids=test.PassengerId.to_numpy(str)
    path=CACHE/'test.npz'
    np.savez_compressed(path,passenger=ids,p=p,chosen=choice,sham=sham)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['p'],p)
    old=np.column_stack([test_decisions(key,ids)[0] for key in RELEASES.values()])
    parent,digest=test_decisions(LATEST,ids)
    common=(old==parent[:,None]).all(1)
    record,hidden,control=(p[:,j]>=.5 for j in range(3))
    first=(record!=parent)&(hidden==parent)&common
    both=first&(control!=parent)
    report=dict(rows=len(ids),parent=LATEST,parent_sha256=digest,eleven_agree=int(common.sum()),
                field_required=int(first.sum()),required_only_with_actual_choice=int(both.sum()),
                training_rows=len(y),test_labels_used=0)
    (RESULT.parent/'field_evidence_test_reach.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
