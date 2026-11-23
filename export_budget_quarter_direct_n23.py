"""Export the exact v44/Mitra/budget mixture used in the three archived grouped validations."""
import json
import shutil
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_partition_evidence_n23 import ART, ROOT, write
from error_overlap_diagnostic import load_authenticated
from exaone_validation_data import compact_features
from export_budget_quarter_n23 import NAME as INCUMBENT_EXPORT
from kaggle_workflow import sha256, validate_submission
from spend_simplex_likelihood_n23 import design

VERSION = 'n23_budget_quarter_three_family'
NAME = 'proposal_'+VERSION


def member(name,test):
    directory=ART/name
    attempts=list((ROOT/'reports'/name/'online_attempts').glob('*.json'))
    assert len(attempts)==1
    receipt=json.loads(attempts[0].read_text())
    assert receipt['evaluation']['Status']=='COMPLETE' and sha256(directory/'submission.csv')==receipt['file_sha256']
    soft=pd.read_csv(directory/'test_probabilities.csv',float_precision='round_trip')
    hard=pd.read_csv(directory/'submission.csv')
    for frame in [soft,hard]:np.testing.assert_array_equal(frame.PassengerId,test.PassengerId)
    np.testing.assert_array_equal(soft.probability>=.5,hard.Transported)
    return soft.probability.to_numpy(), sha256(directory/'test_probabilities.csv')


def original():
    inp,_,fold,models,_=load_authenticated()
    with np.load(ART/'nov23_budget_quarter_replay/original.npz') as z:
        for key,value in [('row',inp['dev']),('y',inp['y']),('fold',fold)]:
            np.testing.assert_array_equal(z[key],value)
        base=(models['v44']+models['rawMitra'])/2
        vote=.75*base+.25*z['alternative']>=.5
        h=vote==z['y']
        result={}
        for key,old_vote in [('incumbent_v131',z['ref']),('two_families',base>=.5)]:
            old=old_vote==z['y']
            gain=h.astype(int)-old.astype(int)
            result[key]=dict(fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                             interval_pp=interval(gain,inp['groups']),folds=[int(gain[fold==k].sum()) for k in np.unique(fold)])
    return dict(accuracy=float(h.mean()),controls=result)


def export():
    check=original()
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    x=design(compact_features(test))[0]
    source=ART/INCUMBENT_EXPORT
    verified=json.loads((ROOT/'reports'/INCUMBENT_EXPORT/'manifest.json').read_text())
    assert sha256(source/'budget.cbm')==verified['model_sha256']
    out=ART/NAME
    out.mkdir(exist_ok=True)
    model=out/'budget.cbm'
    shutil.copyfile(source/'budget.cbm',model)
    v44,h1=member('proposal_batch_v44',test)
    mitra,h2=member('proposal_batch_mitra2',test)
    def repeat(path):
        alt=CatBoostClassifier().load_model(str(path)).predict_proba(x,thread_count=2)[:,1]
        return .75*((v44+mitra)/2)+.25*alt
    p=repeat(model)
    q=repeat(source/'budget.cbm')
    np.testing.assert_array_equal(p,q)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=q)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(out/'submission.csv',index=False)
    validation=validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    controls={}
    changes={}
    for a in (ROOT/'reports').glob('*/online_attempts/*.json'):
        rec=json.loads(a.read_text())
        assert rec.get('file_sha256')!=validation['sha256'],a
        csv=ART/a.parent.parent.name/'submission.csv'
        if csv.exists() and sha256(csv)==rec.get('file_sha256'):
            past=pd.read_csv(csv)
            np.testing.assert_array_equal(past.PassengerId,test.PassengerId)
            changes[a.parent.parent.name]=int((past.Transported.to_numpy()!=(p>=.5)).sum())
            if np.array_equal(past.Transported,(v44+mitra)/2>=.5):
                controls[a.parent.parent.name]=dict(public_score=float(rec['evaluation']['Public Score']),
                                                      submission_ref=rec['submission_ref'])
    assert min(changes.values())>0
    incumbent=pd.read_csv(source/'submission.csv')
    np.testing.assert_array_equal(test.PassengerId,incumbent.PassengerId)
    metrics=dict(**validation,model_sha256=sha256(model),version=VERSION,fit_people=verified['fit_people'],
                 reload_difference=0,source_probabilities={'v44':h1,'v71':h2},matched_previous_csv=len(changes),
                 changed_from_v131=changes['proposal_v131_party_consensus'],nearest_exports=[n for n,v in changes.items() if v==min(changes.values())],
                 minimum_difference=min(changes.values()),changes_from_other_budget_export=int((incumbent.Transported!=(p>=.5)).sum()),
                 confirmed_two_family_uploads=controls,original_random_replay=check)
    for folder in [out,ROOT/'reports'/NAME]:
        write(folder/'manifest.json',metrics)
        write(folder/'selection.json',dict(version=VERSION,threshold=.5))
    print(metrics)


if __name__=='__main__':
    with threadpool_limits(2):export()
