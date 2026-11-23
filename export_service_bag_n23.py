"""Export the service-bag blend and measure coverage against prior releases."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,softmax
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_partition_evidence_n23 import ART,ROOT,write
from broad_service_bag_n23 import design,fit
from export_budget_quarter_n23 import source
from export_personal_bill_mix_n23 import NAME as PERSONAL
from export_three_corrections_n23 import NAME as THREE
from kaggle_workflow import sha256,validate_submission

VERSION='n23_service_bag_quarter'
NAME='proposal_'+VERSION


def export():
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId','Transported'])
    test_path=ROOT/'data/raw/test.csv'
    test=pd.read_csv(test_path,usecols=['PassengerId'])
    frame,groups,size,_=load_roster()
    ids=pd.concat([train.PassengerId,test.PassengerId],ignore_index=True)
    np.testing.assert_array_equal(groups,ids.str.split('_').str[0].astype(int))
    t=np.arange(len(train))
    q=np.arange(len(train),len(frame))
    assert not np.intersect1d(groups[t],groups[q]).size
    frame['BookingSize']=size
    x,g,prior=design(frame)
    alt,params=fit(x,g,prior,train.Transported.to_numpy(bool),t,q,'fixed',return_parameters=True)
    out=ART/NAME
    out.mkdir(exist_ok=True)
    path=out/'service_bag.npz'
    np.savez_compressed(path,weights=params.reshape(x.shape[1:]))
    with np.load(path) as saved:
        logits=np.einsum('nsd,sd->ns',x[q],saved['weights'])
        replay=(softmax(prior[q],axis=1)*expit(logits)).sum(1)
    publish_blend(VERSION,path,test,alt,replay,len(train),(PERSONAL,THREE),'three_candidate_union')


def publish_blend(version,path,test,alt,replay,training_rows,comparators,union_key):
    out=path.parent
    base,_,anchor=source('proposal_v108_party_bills',test)
    _,ref,incumbent=source('proposal_v131_party_consensus',test)
    locked=ref!=(base>=.5)
    def blend(v):
        return np.where(locked,ref,.75*base+.25*v)
    np.testing.assert_array_equal(alt,replay)
    assert np.isfinite(alt).all() and ((alt>=0)&(alt<=1)).all()
    p,r=blend(alt),blend(replay)
    vote=p>=.5
    np.testing.assert_array_equal(vote[locked],ref[locked])
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=p,reload_probability=r)).to_csv(out/'test_probabilities.csv',index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=vote)).to_csv(out/'submission.csv',index=False)
    check=validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
    previous={}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        data=json.loads(receipt.read_text())
        assert data.get('file_sha256')!=check['sha256'],receipt
        name=receipt.parent.parent.name
        original=ART/name/'submission.csv'
        if original.exists() and sha256(original)==data.get('file_sha256'):
            old=pd.read_csv(original)
            np.testing.assert_array_equal(old.PassengerId,test.PassengerId)
            previous[name]=int((old.Transported!=vote).sum())
    assert min(previous.values())>0
    union=vote!=ref
    others={}
    for name in comparators:
        path_csv=ART/name/'submission.csv'
        metadata=json.loads((ROOT/'reports'/name/'manifest.json').read_text())
        assert sha256(path_csv)==metadata['sha256']
        old=pd.read_csv(path_csv)
        np.testing.assert_array_equal(old.PassengerId,test.PassengerId)
        others[name]=int((old.Transported!=vote).sum())
        union|=old.Transported.to_numpy()!=ref
    assert min(others.values())>0
    manifest=dict(**check,version=version,training_rows=training_rows,v108=anchor,v131=incumbent,locked=int(locked.sum()),
                  model_sha256=sha256(path),reload_difference=0,v131_changes=int((vote!=ref).sum()),
                  unchecked_ceiling_correct=3494+int((vote!=ref).sum()),previous_files=len(previous),
                  closest_previous=min(previous.values()),closest_names=[k for k,v in previous.items() if v==min(previous.values())],
                  earlier_candidate_distances=others)
    manifest[union_key]=int(union.sum())
    for folder in (out,ROOT/'reports'/out.name):
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',dict(version=version,threshold=.5))
    print(manifest)


if __name__=='__main__':
    with threadpool_limits(1):
        export()
