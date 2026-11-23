"""Export the service-specific tree with one total unit of training weight per passenger."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_partition_evidence_n23 import ART,ROOT
from broad_service_bag_trees_n23 import expand,fit
from export_service_bag_n23 import PERSONAL,THREE,NAME as LINEAR,publish_blend

VERSION='n23_service_bag_tree_quarter'
NAME='proposal_'+VERSION


def export():
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId','Transported'])
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    frame,groups,size,_=load_roster()
    ids=pd.concat([train.PassengerId,test.PassengerId],ignore_index=True)
    np.testing.assert_array_equal(groups,ids.str.split('_').str[0].astype(int))
    assert not np.intersect1d(groups[:len(train)],groups[len(train):]).size
    frame['BookingSize']=size
    owner,expanded,prior=expand(frame)
    support=owner<len(train)
    query=~support
    labels=train.Transported.to_numpy()[owner[support]]
    tree=fit(expanded[support],labels,prior[support])
    out=ART/NAME
    out.mkdir(exist_ok=True)
    path=out/'service_bag.cbm'
    tree.save_model(str(path))
    def predict(model):
        probability=model.predict_proba(expanded[query],thread_count=2)[:,1]
        return np.bincount(owner[query]-len(train),weights=probability*prior[query],minlength=len(test))
    first=predict(tree)
    second=predict(CatBoostClassifier().load_model(str(path)))
    publish_blend(VERSION,path,test,first,second,len(train),(PERSONAL,THREE,LINEAR),'four_candidate_union')


if __name__=='__main__':
    with threadpool_limits(2):
        export()
