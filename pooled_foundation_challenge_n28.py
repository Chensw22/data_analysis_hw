"""Compare soft and hard corroboration of three foundation models by four raw-data classifiers."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from repeated_archive_misses_n26 import ART, ROOT, count, cohort


def rules(parent,seven,raw):
    sign=np.where(parent,-1.,1.)[:,None]
    foundation=logit(seven[:,1:4].clip(.005,.995))*sign
    independent=raw*sign
    # Each domain gets one vote. Two receipt classifiers share a domain.
    external=np.column_stack([independent[:,0:2].mean(1),independent[:,2:]])
    uni=(foundation>0).all(1)
    maj=(foundation>0).sum(1)>=2
    agreement=(external>0).sum(1)
    average=external.mean(1)
    return {'unanimous_with_native_majority':uni&(agreement>=2),
            'unanimous_with_any_native':uni&(agreement>=1),
            'independent_consensus':maj&(agreement==3),
            'joint_log_odds':maj&(average>0)&(foundation.mean(1)>0),
            'two_banks':((foundation.mean(1)>0)&(average>0)&(agreement>=2)),
            'foundation_unanimity':uni}


def run():
    frame,y,g,f,ids,_,_,seven,versions=inputs();n=len(y)
    parent=versions['random'][:,0]
    wrong=parent[:n]!=y
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    result,models={},{}
    for tag,v in versions.items():
        cols=[]
        for folder,i in [('service_transition_likelihood',1),('receipt_subset_levels',1),('cross_order_texture',-1),('matched_uncertainty_curriculum',1)]:
            with np.load(ART/f'nov28_{folder}/aligned.npz') as z:
                spatial=folder=='cross_order_texture'
                np.testing.assert_array_equal(z['ids' if spatial else 'passenger'],ids)
                np.testing.assert_array_equal(z['fold'],f);np.testing.assert_array_equal(z['y'],y)
                cols.append(z[tag+'_z' if spatial else tag][:,i])
        raw=np.column_stack(cols);models[tag]=raw
        scope={'all':known,'new_orders':known&fresh,'four':known&(v[:n]==parent[:n,None]).all(1)}
        result[tag]={}
        for label,flag in rules(parent,seven,raw).items():
            row={s:count(flag[:n],m,wrong) for s,m in scope.items()}
            row.update(test=int(flag[n:].sum()),test_four=int((flag[n:]&(v[n:]==parent[n:,None]).all(1)).sum()))
            result[tag][label]=row
    np.testing.assert_allclose(models['random'][n:],models['ordered'][n:])
    dest=ART/'nov28_pooled_foundation_challenge';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'aligned.npz',ids=ids,y=y,fold=f,seven=seven,**models)
    (ROOT/'reports/nov23_matched_learning_curve/pooled_foundation_challenge.json').write_text(json.dumps(result,indent=2)+'\n')
    for k in result['random']:
        print(k,[(r[k]['all'],r[k]['four'],r[k]['new_orders']) for r in result.values()],result['random'][k]['test'])


if __name__=='__main__':
    with threadpool_limits(1):run()
