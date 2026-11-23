"""Measure broader, deployable combinations and the true reach of four pending candidates."""
import json
import numpy as np
import pandas as pd
from audit_flip_capacity_n23 import required
from broad_booking_policy_replay_n23 import interval
from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_three_corrections_replay_n23 import scores
from export_budget_quarter_n23 import source
from export_service_bag_n23 import NAME as LINEAR,PERSONAL,THREE
from export_service_bag_tree_n23 import NAME as TREE
from kaggle_workflow import sha256


def rules(base,ref,p):
    # p contains probabilities of four complete candidate blends, with the same overrides.
    fixed=ref!=(base>=.5)
    residual=p-base[:,None]
    distinct=base+residual[:,0]+.5*(residual[:,2]+residual[:,3])
    cumulative=base+residual[:,0]+residual[:,2]+residual[:,3]
    dissent=(p>=.5)!=ref[:,None]
    actions={'all_changes':ref^dissent.any(1),'two_agree':ref^(dissent.sum(1)>=2),
             'distinct_shifts':distinct>=.5,'all_three_shifts':cumulative>=.5}
    return {k:np.where(fixed,ref,v) for k,v in actions.items()}


def test_scope():
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    base,_,_=source('proposal_v108_party_bills',test)
    _,ref,_=source('proposal_v131_party_consensus',test)
    p=[]
    for name in [PERSONAL,THREE,LINEAR,TREE]:
        path=ART/name
        meta=json.loads((ROOT/'reports'/name/'manifest.json').read_text())
        assert sha256(path/'submission.csv')==meta['sha256']
        soft=pd.read_csv(path/'test_probabilities.csv',float_precision='round_trip')
        np.testing.assert_array_equal(soft.PassengerId,test.PassengerId)
        np.testing.assert_array_equal(soft.probability,soft.reload_probability)
        np.testing.assert_array_equal(soft.probability>=.5,pd.read_csv(path/'submission.csv').Transported)
        p.append(soft.probability)
    test_actions=rules(base,ref,np.column_stack(p))
    return {k:int((v!=ref).sum()) for k,v in test_actions.items()}


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        three,ref=scores(frame,g,row,y,seed)
        complete=[three[:,0],three.mean(1)]
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as src:
            base,fold=src['base'],src['fold']
        for folder,key in [('service_bag','fixed'),('service_bag_trees','initial')]:
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for field,v in [('row',row),('y',y),('ref',ref),('fold',fold),('base',base)]:
                    np.testing.assert_array_equal(z[field],v)
                complete.append(np.where(ref==(base>=.5),.75*base+.25*z[key],ref))
        pred=rules(base,ref,np.column_stack(complete))
        hits={k:v==y for k,v in {**pred,'reference':ref}.items()}
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        union=pred['all_changes']!=ref
        old_union=((np.column_stack(complete[:3])>=.5)!=ref[:,None]).any(1)
        fresh=union&~old_union
        reachable=int((union&(ref!=y)).sum())
        need=required(len(y))-int((ref==y).sum())
        plans[seed]=dict(changes=metrics,union=int(union.sum()),oracle_fixed=reachable,gap_even_if_oracle=need-reachable,
                         new_changes=int(fresh.sum()),new_opportunities=int((fresh&(ref!=y)).sum()))
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:(v['fixed'],v['harmed']) for k,v in metrics.items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    changed=test_scope()
    assert changed['all_changes']==214
    write(REPORT/'candidate_composition.json',dict(plans=plans,stable=stable,test_changes=changed))
    print('test changes',changed)


if __name__=='__main__':
    main()
