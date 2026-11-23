"""Locate actual corrections and collateral errors for both pending service candidates."""
import numpy as np
from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def main():
    row,y,frame,_,_=load()
    wrong,right,hard=cohorts(row,y)
    amount=frame[BILLS]
    positive=amount.gt(0).sum(axis=1).to_numpy()
    missing=amount.isna().any(axis=1).to_numpy()
    masks={'zero_positive_complete':(positive==0)&~missing,
           'zero_positive_incomplete':(positive==0)&missing,'one_positive':positive==1,
           'multiple_positive':positive>=2}
    np.testing.assert_array_equal(np.stack(list(masks.values())).sum(0),1)
    plans={}
    for seed in [*map(str,SEEDS),'v131']:
        results={}
        for name,folder,key in [('linear','service_bag','fixed'),('tree','service_bag_trees','initial')]:
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as data:
                np.testing.assert_array_equal(row,data['row'])
                np.testing.assert_array_equal(y,data['y'])
                base,ref,p=data['base'],data['ref'],data[key]
            vote=np.where(ref==(base>=.5),.75*base+.25*p>=.5,ref)
            old=ref==y
            now=vote==y
            stats={k:dict(passengers=int(s.sum()),**paired(now[s],old[s])) for k,s in
                   {**masks,'old_common_wrong':wrong,'old_common_correct':right,'persistent_372':hard}.items()}
            for k in ['fixed','harmed']:
                assert sum(stats[s][k] for s in masks)==paired(now,old)[k]
            results[name]=stats
        plans[seed]=results
    write(REPORT/'service_change_sources.json',plans)
    for k,v in plans['v131'].items():
        print(k,{s:(g['fixed'],g['harmed']) for s,g in v.items()})


if __name__=='__main__':
    main()
