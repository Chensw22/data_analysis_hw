"""Freeze mixture weights by unlabeled test change counts; evaluate them in all local replays."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_booking_canonical_n23 import load_roster
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_service_bag_trees_n23 import expand
from broad_service_pairs_n23 import REPORT,fit_pair,pair_table
from export_service_bag_n23 import PERSONAL,THREE,NAME as LINEAR
from export_service_bag_tree_n23 import NAME as TREE
from export_budget_quarter_n23 import source
from kaggle_workflow import sha256


def full_scores(directory):
    train=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId','Transported'])
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    frame,groups,size,_=load_roster()
    ids=pd.concat([train.PassengerId,test.PassengerId],ignore_index=True)
    np.testing.assert_array_equal(groups,ids.str.split('_').str[0].astype(int))
    t=np.arange(len(train))
    q=np.arange(len(train),len(frame))
    assert not np.intersect1d(groups[t],groups[q]).size
    frame['BookingSize']=size
    owner,rows,prior=expand(frame)
    people,xt,wt=pair_table(owner,rows,prior,t,t)
    model=fit_pair(xt,train.Transported.to_numpy()[people],wt)
    path=directory/'pairs.cbm'
    model.save_model(str(path))
    iq,xq,wq=pair_table(owner,rows,prior,q,q)
    def forecast(fitted):
        p=fitted.predict_proba(xq,thread_count=2)[:,1]
        return np.bincount(iq-len(train),weights=wq*p,minlength=len(test))
    p=forecast(model)
    replay=forecast(CatBoostClassifier().load_model(str(path)))
    np.testing.assert_array_equal(p,replay)
    base,_,anchor=source('proposal_v108_party_bills',test)
    _,ref,incumbent=source('proposal_v131_party_consensus',test)
    np.savez_compressed(directory/'probabilities.npz',id=test.PassengerId.to_numpy(str),base=base,ref=ref,
                        pair=p,reload=replay)
    return test,p,base,ref,dict(model_sha256=sha256(path),anchor=anchor,incumbent=incumbent,training_rows=len(train))


def changed(p,base,ref,w):
    return np.where(ref==(base>=.5),(1-w)*base+w*p>=.5,ref)


def coverage_weights(p,base,ref):
    mask=(ref==(base>=.5))&(ref!=(p>=.5))
    cost=np.sort(np.abs((.5-base[mask])/(p[mask]-base[mask])))
    result={'quarter':.25}
    for limit in (150,250,350):
        if limit>=len(cost):
            continue
        weight=(cost[limit-1]+cost[limit])/2
        assert (changed(p,base,ref,weight)!=ref).sum()==limit
        result['change_'+str(limit)]=float(weight)
    result['full']=1.
    return result


def main():
    directory=ART/'nov23_service_pair_full'
    directory.mkdir(exist_ok=True)
    test,p,anchor,incumbent,metadata=full_scores(directory)
    weights=coverage_weights(p,anchor,incumbent)
    previous={}
    for n in (PERSONAL,THREE,LINEAR,TREE):
        file=ART/n/'submission.csv'
        assert sha256(file)==json.loads((ROOT/'reports'/n/'manifest.json').read_text())['sha256']
        other=pd.read_csv(file)
        np.testing.assert_array_equal(other.PassengerId,test.PassengerId)
        previous[n]=other.Transported.to_numpy()
    earlier=np.column_stack(list(previous.values()))
    old_union=(earlier!=incumbent[:,None]).any(1)
    test_info={}
    for k,w in weights.items():
        v=changed(p,anchor,incumbent,w)
        difference=v!=incumbent
        test_info[k]=dict(weight=w,changes=int(difference.sum()),new_changes=int((difference&~old_union).sum()),
                          distance_from_pending={name:int((v!=old).sum()) for name,old in previous.items()})
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_service_pairs/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            base,ref,alt=z['base'],z['ref'],z['same_passenger_both']
        hits={k:changed(alt,base,ref,w)==y for k,w in weights.items()}
        hits['reference']=ref==y
        plans[seed]={k:paired(v,ref==y) for k,v in hits.items()}
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'test_budget.json',dict(training=metadata,plans=plans,stable=stable,test=test_info))
    for k,v in test_info.items():
        print(k,v['changes'],v['new_changes'],[d[k]['fixed']-d[k]['harmed'] for d in plans.values()])


if __name__=='__main__':
    with threadpool_limits(2):
        main()
