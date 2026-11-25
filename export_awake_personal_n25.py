"""Train the awake-only TabICL context and freeze its direct edits against the actual nine-vote release."""
import json
import numpy as np
import pandas as pd
import torch

from awake_context_ranks_n25 import OUT as CHECKS
from exaone_validation_data import CATEGORICAL,compact_features
from export_budget_quarter_n23 import ART,ROOT,source
from full_roster_booking_validation_n23 import data
from kaggle_workflow import sha256,validate_submission
from relational_error_corrector import write
from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer

VERSION='n25_awake_personal_tabicl'
NAME='proposal_'+VERSION
SEED=891


def run():
    assert sha256(WEIGHT)==DIGEST
    torch.set_num_threads(2)
    train,y,booking,fold=data()
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=lambda c:c!='Name')
    test_booking=test.PassengerId.str.split('_').str[0]
    assert not np.isin(test_booking,booking).any()
    roster=pd.concat([train,test],ignore_index=True)
    x=compact_features(roster)
    awake_train=train.CryoSleep.eq(False).to_numpy()
    awake_test=test.CryoSleep.eq(False).to_numpy()
    with np.load(CHECKS/'0_personal.npz') as z:
        t,q=z['t'],z['q']
        np.testing.assert_array_equal(t,np.flatnonzero(awake_train&(fold!=0)))
        np.testing.assert_array_equal(q,np.flatnonzero(awake_train&(fold==0)))
        np.testing.assert_array_equal(z['fields'],x.columns)
        np.testing.assert_array_equal(z['train_y'],y[t])
        np.testing.assert_array_equal(z['p'],infer(x,y,t,q,SEED))
    training=np.flatnonzero(awake_train)
    query=np.flatnonzero(awake_test)+len(train)
    out=ART/NAME
    out.mkdir(exist_ok=True)
    context_file=out/'training_context.csv'
    context=x.iloc[training].reset_index(drop=True)
    labels=y[training]
    context.assign(Transported=labels).to_csv(context_file,index=False)
    saved=pd.read_csv(context_file,dtype={k:str for k in CATEGORICAL},float_precision='round_trip')
    restored_y=saved.pop('Transported').to_numpy()
    pd.testing.assert_frame_equal(context,saved,check_dtype=False)
    np.testing.assert_array_equal(labels,restored_y)
    query_x=x.iloc[query].reset_index(drop=True)
    parents={}
    _,native,parents['native']=source('proposal_online_first_counts15',test)
    _,prior,parents['v131']=source('proposal_v131_party_consensus',test)
    nine_p,nine,parents['nine']=source('proposal_n24_selected_dissent',test)
    allow=(prior==native)&(nine==prior)&awake_test
    def apply_predictions(p):
        candidate=nine_p.copy()
        candidate[allow]=p[allow[awake_test]]
        return candidate
    p=apply_predictions(infer(x,y,training,query,SEED))
    replay_frame=pd.concat([saved,query_x],ignore_index=True)
    t=np.arange(len(saved))
    q=np.arange(len(saved),len(replay_frame))
    replay=apply_predictions(infer(replay_frame,restored_y,t,q,SEED))
    np.testing.assert_array_equal(replay,p)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    prediction=p>=.5
    changed=prediction!=nine
    assert changed.any() and not (changed&~allow).any()
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=prediction)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',test_path)
    distances={}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        r=json.loads(receipt.read_text())
        if r.get('evaluation',{}).get('Status')!='COMPLETE':
            continue
        name=receipt.parents[1].name
        path=ART/name/'submission.csv'
        assert sha256(path)==r['file_sha256']
        other=pd.read_csv(path)
        np.testing.assert_array_equal(other.PassengerId,test.PassengerId)
        distances[name]=int((other.Transported!=prediction).sum())
    closest=min(distances.values())
    first='proposal_n25_confirmed_random_fields'
    other=pd.read_csv(ART/first/'submission.csv')
    np.testing.assert_array_equal(other.PassengerId,test.PassengerId)
    assert sha256(ART/first/'submission.csv')==json.loads((ROOT/'reports'/first/'manifest.json').read_text())['sha256']
    manifest=dict(**check,version=VERSION,parents=parents,model_sha256=sha256(context_file),
                  pretrained_checkpoint_sha256=DIGEST,reload_difference=0,training_rows=len(t),
                  source_train_sha256=sha256(ROOT/'data/raw/train.csv'),source_test_sha256=sha256(test_path),
                  eligible_test=int(allow.sum()),nine_changes=int(changed.sum()),
                  changed_to_true=int((changed&prediction).sum()),changed_to_false=int((changed&~prediction).sum()),
                  completed_exports=len(distances),closest_submitted_changes=closest,
                  nearest_submissions=sorted(n for n,d in distances.items() if d==closest),
                  difference_from_queued=int((prediction!=other.Transported).sum()),
                  overlap_with_queued_edits=int((changed&(other.Transported!=nine)).sum()))
    selection=dict(version=VERSION,model='TabICLv2',mode='personal/direct',seed=SEED,context='CryoSleep == False',
                   fields=list(x.columns),evidence='reports/nov23_matched_learning_curve/awake_context_ranks.json')
    for directory in (out,ROOT/'reports'/NAME):
        write(directory/'manifest.json',manifest)
        write(directory/'selection.json',selection)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    run()
