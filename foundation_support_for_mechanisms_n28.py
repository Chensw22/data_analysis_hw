"""Evaluate new raw-model corrections with independently trained foundation opinions."""
from pathlib import Path
import json

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import PARENTS,inputs
from repeated_archive_misses_n26 import ART,ROOT,cohort,count


def views(tag,ids,y,fold,margin,parent):
    opinions=[]
    for name,i,j in [('service_transition_likelihood',1,0),('receipt_subset_levels',1,0),
                     ('cross_order_texture',-1,1),('matched_uncertainty_curriculum',1,0)]:
        spatial=name=='cross_order_texture'
        with np.load(ART/f'nov28_{name}/aligned.npz') as z:
            np.testing.assert_array_equal(z['ids' if spatial else 'passenger'],ids)
            np.testing.assert_array_equal(z['fold'],fold)
            np.testing.assert_array_equal(z['y'],y)
            raw=z[tag+'_z' if spatial else tag]
        opinions.append(((raw[:,i]>=0)!=parent)&((margin+raw[:,i]-raw[:,j]>=0)!=parent))
    return np.column_stack(opinions)


def decisions(parent,versions,raw,seven):
    dissent=(versions!=parent[:,None]).sum(1)
    votes=((seven>=.5)!=parent[:,None])
    raw_families=(raw[:,0]|raw[:,1]).astype(int)+raw[:,2]+raw[:,3]
    foundations=votes[:,1:4].sum(1)
    majority=dissent>=2
    return {'tied_vote':majority,
            'any_tied_and_raw':(dissent>0)&(raw_families>=1),
            'foundation_and_two_raw':(foundations>=2)&(raw_families>=2),
            'unanimous_foundations_and_raw':(foundations==3)&(raw_families>=1),
            'raw_or_tied_with_foundations':(foundations>=2)&((dissent>0)|(raw_families>=2)),
            'majority_plus_foundation_raw':majority|((foundations==3)&(raw_families>=1)),
            'majority_plus_two_raw':majority|((foundations>=2)&(raw_families>=2))}


def run():
    frame,y,g,f,ids,_,_,seven,v=inputs();n=len(y)
    parent=v['random'][:,0]
    margin=np.where(parent,1.,-1.)*abs(logit(np.median(seven[:,1:4],axis=1).clip(.002,.998)))
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    wrong=parent[:n]!=y
    report={};arrays={}
    for tag,versions in v.items():
        raw=views(tag,ids,y,f,margin,parent)
        scope={'all':known,'new_orders':known&fresh,'four':known&(versions[:n]==parent[:n,None]).all(1)}
        scope['four_seven']=scope['four']&((seven[:n]>=.5)==parent[:n,None]).all(1)
        report[tag]={}
        for name,flag in decisions(parent,versions,raw,seven).items():
            row={s:count(flag[:n],m,wrong) for s,m in scope.items()}
            row.update(test=int(flag[n:].sum()),test_four=int((flag[n:]&(versions[n:]==parent[n:,None]).all(1)).sum()))
            report[tag][name]=row
        arrays[tag]=raw
    np.testing.assert_array_equal(arrays['random'][n:],arrays['ordered'][n:])
    dest=ART/'nov28_foundation_support_for_mechanisms';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',ids=ids,y=y,fold=f,seven=seven,**arrays)
    (ROOT/'reports/nov23_matched_learning_curve/foundation_support_for_mechanisms.json').write_text(json.dumps(report,indent=2)+'\n')
    for k in report['random']:
        print(k,*[(r[k]['all'],r[k]['four'],r[k]['new_orders']) for r in report.values()],report['random'][k]['test'],report['random'][k]['test_four'])


if __name__=='__main__':
    with threadpool_limits(1):run()
