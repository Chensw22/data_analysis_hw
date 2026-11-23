"""Export two locally supported controls for the completed joint-spending evaluation."""
import json
import shutil

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from broad_joint_observed_state_n23 import predict, views
from broad_partition_evidence_n23 import ART, ROOT, write
from exaone_validation_data import compact_features, CATEGORICAL
from export_budget_quarter_n23 import source
from export_joint_basket_n23 import NAME as JOINT_BASKET
from kaggle_workflow import sha256, validate_submission
from tabicl_routed_context_n23 import DIGEST, WEIGHT

VERSIONS = {'direct': 'n23_direct_basket_quarter', 'paired': 'n23_sleep_basket_joint_quarter'}


def evaluate(frame, y, base, ref):
    t, q = np.arange(len(y)), np.arange(len(y), len(frame))
    table = views(frame)
    probs = {}
    for name, (x, state) in table.items():
        target = np.zeros(len(frame), int)
        target[t] = 2*state[t]+y
        joint = predict(x, target, t, q, 0).reshape(len(q),3,2)
        selected = joint[np.arange(len(q)), state[q]]
        probs[name] = selected[:,1]/selected.sum(1)
        if name == 'basket':
            probs['direct'] = predict(x.assign(observed_state=state.astype(str)), y, t, q, 0)[:,1]
    probs['paired'] = (probs['sleep'] + probs['basket']) / 2
    return {k:np.where((base >= .5) == ref,.75*base+.25*v,ref.astype(float)) for k,v in probs.items()}


def main():
    assert sha256(WEIGHT) == DIGEST
    train, test = (pd.read_csv(ROOT/f'data/raw/{s}.csv') for s in ('train','test'))
    frame = compact_features(pd.concat([train,test],ignore_index=True))
    y = train.Transported.to_numpy(int)
    base,_,bhash = source('proposal_v108_party_bills',test)
    _,ref,rhash = source('proposal_v131_party_consensus',test)
    previous, previous_vote, _ = source(JOINT_BASKET, test)
    initial = evaluate(frame,y,base,ref)
    np.testing.assert_array_equal(initial['basket'] >= .5, previous_vote)
    np.testing.assert_allclose(initial['basket'],previous,rtol=0,atol=1e-12)
    first = ART/('proposal_'+VERSIONS['direct'])
    first.mkdir(exist_ok=True)
    context = first/'training_context.csv'
    frame.iloc[:len(train)].assign(transported=y).to_csv(context,index=False)
    restored = pd.read_csv(context,dtype={k:str for k in CATEGORICAL},float_precision='round_trip')
    state = restored.pop('transported').to_numpy()
    pd.testing.assert_frame_equal(restored,frame.iloc[:len(train)])
    np.testing.assert_array_equal(y,state)
    compared = evaluate(pd.concat([restored,frame.iloc[len(y):]],ignore_index=True),state,base,ref)
    for k,version in VERSIONS.items():
        name='proposal_'+version
        out=ART/name
        out.mkdir(exist_ok=True)
        if out!=first:
            shutil.copyfile(context,out/'training_context.csv')
        p,reload=initial[k],compared[k]
        np.testing.assert_array_equal(p,reload)
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=reload)).to_csv(out/'test_probabilities.csv',index=False)
        csv=out/'submission.csv'
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(csv,index=False)
        checked=validate_submission(csv,ROOT/'data/raw/test.csv')
        closest={}
        for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
            rec=json.loads(path.read_text())
            if path.parents[1].name==name:continue
            assert rec.get('file_sha256')!=checked['sha256']
            other=ART/path.parents[1].name/'submission.csv'
            if other.is_file() and sha256(other)==rec.get('file_sha256'):
                old=pd.read_csv(other)
                np.testing.assert_array_equal(test.PassengerId,old.PassengerId)
                closest[path.parents[1].name]=int((old.Transported.to_numpy()!=(p>=.5)).sum())
        gap=min(closest.values())
        assert gap>0
        manifest=dict(**checked,version=version,model_sha256=sha256(out/'training_context.csv'),checkpoint_sha256=DIGEST,
                      reload_difference=0,context_passengers=len(y),anchors=[bhash,rhash],changed_from_v131=int(((p>=.5)!=ref).sum()),
                      changed_from_joint_basket=int(((p>=.5)!=previous_vote).sum()),distance_between_comparators=int(((initial['direct']>=.5)!=(initial['paired']>=.5)).sum()),
                      closest_previous={m:v for m,v in closest.items() if v==gap})
        for folder in (out,ROOT/'reports'/name):
            write(folder/'manifest.json',manifest)
            write(folder/'selection.json',dict(version=version,sha256=checked['sha256']))
        print(name,manifest,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):main()
