"""Compare full-strength roster-wide alternatives to a local edit of the incumbent."""
import numpy as np
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_opposing_passengers_n23 import REPORT
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def views(seed,row,y,fold):
    result={}
    sources={'personal':('two_source_manifest','direct'),'foundation':('pretrained_spending_views','budget')}
    for k,(folder,name) in sources.items():
        with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
            for field,expected in [('row',row),('y',y),('fold',fold)]:
                np.testing.assert_array_equal(z[field],expected)
            result[k]=z[name]
    if seed=='v131':
        tree_path=ART/'nov23_budget_quarter_replay/original.npz'
        field='alternative'
    else:
        tree_path=ART/f'nov23_simplex_likelihood/{seed}.npz'
        field='budget_prior'
    with np.load(tree_path) as z:
        for key,expected in [('row',row),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[key],expected)
        result['budget_tree']=z[field]
    return result


def combine(p):
    return {**p,'personal_foundation':(p['personal']+p['foundation'])/2,
            'three_views':(p['personal']+p['foundation']+p['budget_tree'])/3}


def main():
    row,y,_,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,ref,base=z['fold'],z['ref'],z['base']
        scores=combine(views(seed,row,y,fold))
        hit={'reference':ref==y}
        for k,p in scores.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            hit[k]=np.where(ref==(base>=.5),p>=.5,ref)==y
        plans[seed]={k:paired(v,ref==y) for k,v in hit.items()}
        if seed!='v131':
            for k,v in hit.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:(v['fixed'],v['harmed']) for k,v in plans[seed].items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'replacement_views.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
