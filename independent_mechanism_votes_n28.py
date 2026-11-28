"""Compare four new incremental opinions where the four score-tied submissions agree."""
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from repeated_archive_misses_n26 import ART, ROOT, cohort, count


def run():
    frame,y,g,fold,ids,_,_,seven,versions=inputs()
    parent=versions['random'][:,0]
    n=len(y)
    wrong=parent[:n]!=y
    _,_,known,fresh=cohort(frame.iloc[:n],y,fold)
    margin=np.where(parent,1.,-1.)*abs(logit(np.median(seven[:,1:4],axis=1).clip(.002,.998)))
    studies=[('matched_uncertainty_curriculum',1,0),('service_transition_likelihood',1,0),
             ('receipt_subset_levels',1,0),('cross_order_texture',-1,1)]
    report,archives={},{}
    for tag,parents in versions.items():
        proposals=[]
        for directory,i,j in studies:
            with np.load(ART/f'nov28_{directory}/aligned.npz') as z:
                spatial=directory=='cross_order_texture'
                np.testing.assert_array_equal(z['ids' if spatial else 'passenger'],ids)
                np.testing.assert_array_equal(z['y'],y)
                np.testing.assert_array_equal(z['fold'],fold)
                scores=z[tag+'_z' if spatial else tag]
            proposals.append(((scores[:,i]>=0)!=parent)&((margin+scores[:,i]-scores[:,j]>=0)!=parent))
        votes=np.column_stack(proposals)
        tied=(parents!=parent[:,None]).sum(1)
        same7=((seven>=.5)==parent[:,None]).all(1)
        # The two receipt models count as one family. The other families use
        # training difficulty and adjacent-order measurements, respectively.
        families=votes[:,0].astype(int)+(votes[:,1]|votes[:,2])+votes[:,3]
        rules={'two_families':families>=2,'three_families':families==3,
               'four_model_vote':tied>=2,'any_tie_with_raw':(tied>0)&(families>=1),
               'majority_or_two_families':(tied>=2)|(families>=2),
               'one_reference_plus_raw_or_two_independent':((tied>0)&(families>0))|(families>=2)}
        report[tag]={}
        for label,flag in rules.items():
            report[tag][label]={'all':count(flag[:n],known,wrong),'new_orders':count(flag[:n],known&fresh,wrong),
                'four':count(flag[:n],known&(tied[:n]==0),wrong),
                'four_seven':count(flag[:n],known&(tied[:n]==0)&same7[:n],wrong),'test':int(flag[n:].sum()),
                'test_four':int((flag[n:]&(tied[n:]==0)).sum())}
            archives[tag+'_'+label]=flag
    dest=ART/'nov28_independent_mechanism_votes';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',passenger=ids,y=y,fold=fold,**archives)
    (ROOT/'reports/nov23_matched_learning_curve/independent_mechanism_votes.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    with threadpool_limits(1):run()
