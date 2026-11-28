"""Validate location rankings using independent passenger-level evidence on exact profile matches."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from repeated_archive_misses_n26 import ART,ROOT,cohort,count


def run():
    frame,y,g,f,ids,_,_,seven,versions=inputs();n=len(y)
    parent=versions['random'][:,0]
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    wrong=parent[:n]!=y
    margin=np.where(parent,1.,-1.)*abs(logit(np.median(seven[:,1:4],axis=1).clip(.002,.998)))
    foundation=((seven[:,1:4]>=.5)!=parent[:,None]).sum(1)
    report={};saved={}
    for tag,v in versions.items():
        sources=[]
        for key,folder in [('ids','nov28_exact_profile_location_rank'),('ids','nov28_pooled_foundation_challenge'),
                           ('ids','nov28_foundation_support_for_mechanisms')]:
            with np.load(ART/folder/'aligned.npz') as z:
                np.testing.assert_array_equal(z[key],ids);np.testing.assert_array_equal(z['fold'],f)
                np.testing.assert_array_equal(z['y'],y)
                sources.append(z[tag])
        loc,pooled,raw=sources
        absolute,diff,count_ref=loc.T
        matched=(count_ref>0)&((absolute>=0)!=parent)
        flag=matched&((margin+diff>=0)!=parent)
        direct=(pooled>=0)!=parent[:,None]
        # Both billing views are derived from the receipt; count them as one.
        support=(direct[:,:2].all(1)).astype(int)+direct[:,2]+direct[:,3]
        increments=(raw[:,:2].any(1)).astype(int)+raw[:,2]+raw[:,3]
        strict=flag&(foundation>=2)
        policies={'native_raw':flag&(support>=2),'novel_raw':flag&(increments>=1),
                  'any_foundation':flag&(foundation>=1),'foundation_majority':strict,
                  'foundation_with_raw':matched&(foundation>=2)&(support>=2)}
        same=(v==parent[:,None]).all(1);same7=((seven>=.5)==parent[:,None]).all(1)
        scope={'all':known,'new_orders':known&fresh,'four':known&same[:n],'four_seven':known&same[:n]&same7[:n]}
        report[tag]={}
        for rule,keep in policies.items():
            out={s:count(keep[:n],m,wrong) for s,m in scope.items()}
            out.update(test=int(keep[n:].sum()),test_four=int((keep[n:]&same[n:]).sum()))
            report[tag][rule]=out;saved[tag+'_'+rule]=keep
    dest=ART/'nov28_confirm_exact_profile_locations';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',ids=ids,y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/confirm_exact_profile_locations.json').write_text(json.dumps(report,indent=2)+'\n')
    for k in report['random']:
        print(k,[(r[k]['all'],r[k]['four'],r[k]['four_seven'],r[k]['new_orders']) for r in report.values()],report['random'][k]['test'])


if __name__=='__main__':
    with threadpool_limits(1):run()
