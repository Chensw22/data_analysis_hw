"""Export frozen TabICL conditional on the observed spending category above v131."""
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from broad_joint_observed_state_n23 import predict, views
from broad_partition_evidence_n23 import ART, ROOT, write
from exaone_validation_data import compact_features, CATEGORICAL
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256, validate_submission
from tabicl_routed_context_n23 import DIGEST, WEIGHT

VERSION='n23_joint_basket_quarter'
NAME='proposal_'+VERSION


def main():
    assert sha256(WEIGHT)==DIGEST
    train=pd.read_csv(ROOT/'data/raw/train.csv')
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    assert not set(train.PassengerId.str.split('_').str[0]) & set(test.PassengerId.str.split('_').str[0])
    table,state=views(compact_features(pd.concat([train,test],ignore_index=True)))['basket']
    n=len(train)
    y=train.Transported.to_numpy(int)
    t=np.arange(n)
    q=np.arange(n,len(table))
    out=ART/NAME
    out.mkdir(exist_ok=True)
    context=out/'training_context.csv'
    table.iloc[t].assign(observed_state=state[t],transported=y).to_csv(context,index=False)
    saved=pd.read_csv(context,dtype={k:str for k in CATEGORICAL},float_precision='round_trip')
    restored_y=saved.pop('transported').to_numpy()
    restored_state=saved.pop('observed_state').to_numpy()
    np.testing.assert_array_equal(state[t],restored_state)
    np.testing.assert_array_equal(y,restored_y)
    pd.testing.assert_frame_equal(saved,table.iloc[t])
    base,_,b_hash=source('proposal_v108_party_bills',test)
    _,ref,r_hash=source('proposal_v131_party_consensus',test)
    def infer(full, ys, observed_state):
        # Query placeholders are never included in the labeled training context.
        target=np.zeros(len(full),int)
        target[t]=2*observed_state[t]+ys
        joint=predict(full,target,t,q,0).reshape(len(q),3,2)
        slice_prob=joint[np.arange(len(q)),observed_state[q]]
        assert np.all(slice_prob.sum(1)>0)
        conditional=slice_prob[:,1]/slice_prob.sum(1)
        return np.where((base>=.5)==ref,.75*base+.25*conditional,ref.astype(float))
    p=infer(table,y,state)
    reload=infer(pd.concat([saved,table.iloc[q]],ignore_index=True),restored_y,np.r_[restored_state,state[q]])
    np.testing.assert_array_equal(p,reload)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=reload)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=p>=.5)).to_csv(out/'submission.csv',index=False)
    checked=validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    manifest=dict(**checked,version=VERSION,checkpoint_sha256=DIGEST,model_sha256=sha256(context),reload_difference=0,
                  context_passengers=n,changed_from_v131=int(((p>=.5)!=ref).sum()),
                  v108_sha256=b_hash,v131_sha256=r_hash)
    for folder in (out,ROOT/'reports'/NAME):
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',dict(version=VERSION,sha256=checked['sha256']))
    print(manifest,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):main()
