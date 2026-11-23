"""Count new repeatable rescues and collateral errors beyond the previous seventy-five."""
import numpy as np

from broad_bill_neighbor_decision_n23 import opinions
from broad_fixed_vote_totals_n23 import REPORT
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,write
from broad_tree_field_dividends_n23 import hard_cohorts


def main():
    row,y,frame,_,_=load()
    _,_,hard,correct=hard_cohorts(row,y)
    history={}
    for seed in map(str,SEEDS):
        fold,_,ref,bills,near=opinions(seed,row,y,frame.BookingSize.to_numpy()>1)
        def read(folder,keys):
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as data:
                for k,expected in [('fold',fold),('ref',ref),('row',row),('y',y)]:
                    np.testing.assert_array_equal(data[k],expected)
                return np.column_stack([data[k] for k in keys])
        previous=np.column_stack([bills,near,read('abstaining_fields',['personal_first','bill_first','averaged_order']),
                                  read('dependence_anchors',['independent','single','double']),
                                  read('opposing_passengers',['matched_conditional'])])
        fresh={'service':read('service_bag',['pooled_logit','fixed','learned']),
               'service_trees':read('service_bag_trees',['initial','mixture']),
               'subspace':read('subspace_discriminants',['diagonal_average','qda_average']),
               'codes':read('known_manifest_codes',['shared_binary','code_mean','code_conditional','code_marginal',
                                                    'ordinary_conditional','ordinary_marginal'])}
        for k,p in {'previous':previous,**fresh,'combined':np.column_stack(list(fresh.values()))}.items():
            history.setdefault(k,[]).append((p>=.5)==y[:,None])
    fixed={k:np.all(parts,axis=0).any(1)&hard for k,parts in history.items()}
    harmed={k:~np.any(parts,axis=0) for k,parts in history.items()}
    harmed={k:v.any(1)&correct for k,v in harmed.items()}
    assert fixed['previous'].sum()==75 and harmed['previous'].sum()==370
    result={k:dict(fixed=int(v.sum()),novel_fixed=int((v&~fixed['previous']).sum()),
                   harmed=int(harmed[k].sum()),novel_harmed=int((harmed[k]&~harmed['previous']).sum()))
            for k,v in fixed.items()}
    write(REPORT/'rescue_overlap.json',result)
    print(result)


if __name__=='__main__':
    main()
