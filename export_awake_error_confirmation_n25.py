"""Freeze an error-trained TabICL filter on top of the previously verified awake candidate."""
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit,logit

from awake_error_demonstrations_n25 import OUT as FOLDS,error_context
from exaone_validation_data import compact_features
from export_awake_personal_n25 import NAME as AWAKE
from export_budget_quarter_n23 import ART,ROOT,source
from export_party_bill import TEACHER
from full_roster_booking_validation_n23 import data
from grouped_peer_teacher_n25 import OUT as REFERENCE
from kaggle_workflow import sha256,validate_submission
from relational_error_corrector import write
from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer

VERSION='n25_awake_error_confirmation'
NAME='proposal_'+VERSION


def fit_context(x,y,awake,teacher,training,seed):
    train=np.flatnonzero(awake&training)
    side=teacher>=.5
    wrong=y!=side
    t,fractions=error_context(train,wrong,side,np.random.default_rng(seed))
    return dict(frame=x.iloc[t].assign(reference_probability=teacher[t],reference_class=side[t].astype(float)).reset_index(drop=True),
                failed=wrong[t],rows=t,correct_fraction=fractions)


def infer_failure(pack,frame,probability,seed):
    query=frame.assign(reference_probability=probability,reference_class=(probability>=.5).astype(float))
    whole=pd.concat([pack['frame'],query],ignore_index=True)
    pt=np.arange(len(pack['frame']))
    pq=np.arange(len(pt),len(whole))
    p=infer(whole,pack['failed'],pt,pq,seed)
    return p,expit(logit(p.clip(.00001,.99999))+np.log(pack['correct_fraction'][(probability>=.5).astype(int)]))


def run():
    torch.set_num_threads(2)
    assert sha256(WEIGHT)==DIGEST
    train,y,g,fold=data()
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=lambda c:c!='Name')
    assert not np.isin(g,test.PassengerId.str.split('_').str[0]).any()
    x=compact_features(pd.concat([train,test],ignore_index=True))
    awake=train.CryoSleep.eq(False).to_numpy()
    awake_test=test.CryoSleep.eq(False).to_numpy()
    xtrain,xtest=x.iloc[:len(y)],x.iloc[len(y):].reset_index(drop=True)
    teacher=np.empty(len(y))
    with np.load(REFERENCE/'outer0.npz') as z:
        teacher[z['t']]=z['p']
    with np.load(REFERENCE/'release0.npz') as z:
        teacher[z['q']]=z['native']
    context=fit_context(xtrain,y,awake,teacher,fold!=0,942)
    q=np.flatnonzero((fold==0)&awake)
    first,_=infer_failure(context,xtrain.iloc[q],teacher[q],891)
    with np.load(FOLDS/'0_balanced_error.npz') as z:
        for key,val in [('p',first),('t',context['rows']),('q',q),('training_correct_fraction',context['correct_fraction'])]:
            np.testing.assert_array_equal(z[key],val)
    top=json.loads((TEACHER/'manifest.json').read_text())['raw_component']['files']['raw_probabilities.npz']
    assert sha256(TEACHER/'raw_probabilities.npz')==top
    with np.load(TEACHER/'raw_probabilities.npz') as z:
        np.testing.assert_array_equal(z['train_passenger_ids'],train.PassengerId)
        teacher=z['oof_raw'].astype(float)
    assert len(teacher)==len(y) and np.isfinite(teacher).all()
    pack=fit_context(xtrain,y,awake,teacher,np.ones(len(y),bool),942)
    native,base,native_sha=source('proposal_online_first_counts15',test)
    nine_p,nine,nine_sha=source('proposal_n24_selected_dissent',test)
    ap=ART/AWAKE
    queued=pd.read_csv(ap/'test_probabilities.csv',float_precision='round_trip')
    labels=pd.read_csv(ap/'submission.csv')
    manifest=json.loads((ROOT/'reports'/AWAKE/'manifest.json').read_text())
    assert sha256(ap/'submission.csv')==manifest['sha256']
    np.testing.assert_array_equal(queued.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(labels.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(queued.probability,queued.reload_probability)
    np.testing.assert_array_equal(queued.probability>=.5,labels.Transported)
    neural=labels.Transported.to_numpy()
    changes=neural!=nine
    assert changes.sum()==135 and not (changes&(~awake_test|(base!=nine))).any()
    out=ART/NAME
    out.mkdir(exist_ok=True)
    model=out/'error_context.joblib'
    joblib.dump(pack,model)
    def predict(context):
        _,error=infer_failure(context,xtest.loc[awake_test],native[awake_test],891)
        outcome=np.where(base[awake_test],1-error,error)>=.5
        confirm=np.zeros(len(test),bool)
        confirm[awake_test]=outcome==neural[awake_test]
        return np.where(changes&confirm,queued.probability,nine_p)
    p=predict(pack)
    replay=predict(joblib.load(model))
    np.testing.assert_array_equal(p,replay)
    hard=p>=.5
    accepted=hard!=nine
    assert accepted.any() and not (accepted&~changes).any()
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=hard)).to_csv(out/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=replay)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',test_path)
    receipts=[json.loads(path.read_text()) for path in (ROOT/'reports').glob('*/online_attempts/*.json')]
    assert all(r['file_sha256']!=check['sha256'] for r in receipts)
    for other in ('proposal_n25_confirmed_random_fields',AWAKE,'proposal_n25_confirmed_awake_union'):
        assert sha256(ART/other/'submission.csv')!=check['sha256']
    output=dict(**check,version=VERSION,model_sha256=sha256(model),reload_difference=0,training_rows=len(pack['rows']),
                pretrained_checkpoint_sha256=DIGEST,teacher_sha256=top,
                parents=dict(nine=nine_sha,native=native_sha,awake=manifest['sha256']),assessed_edits=int(changes.sum()),
                nine_changes=int(accepted.sum()),changed_to_true=int((accepted&hard).sum()),changed_to_false=int((accepted&~hard).sum()))
    recipe=dict(version=VERSION,mode='full_confirmed_by_balanced_error',model_seed=891,context_sampling_seed=942,
                evidence='reports/nov23_matched_learning_curve/awake_error_demonstrations.json')
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',output)
        write(folder/'selection.json',recipe)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    run()
