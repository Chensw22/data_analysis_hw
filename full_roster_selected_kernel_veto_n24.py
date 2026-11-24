"""Refit both predictors in the best single-region two-vote rule on two new booking partitions."""
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_error_partition_kernel_n24 import tree
from full_roster_mistake_influence_n24 import design
from full_roster_selected_tree_replay_n24 import refit,original_prediction


def kernel(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    z = ((x-x[t].mean(0))/x[t].std(0).clip(.2)).clip(-7,7)
    return tree(z,y,g,t,q,'random',2,0)


def evaluate(trees,kernels,base,y,g,f,subsets):
    pred = np.where((kernels>=.5)==base,trees>=.5,base)
    return {key:stats(pred[ix],base[ix],y[ix],g[ix],f[ix]) for key,ix in subsets.items()}


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,_,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/archive_scores.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold),('row',rows),('old_fold',of)]:
            np.testing.assert_array_equal(z[k],v)
        j = z['names'].tolist().index('error_partition_kernel/rand')
        first,past = z['full'][:,j],z['historical'][:,j]
    ix = np.arange(len(y))
    orig = original_prediction(ix,fold,g,y,'full')
    old_tree = original_prediction(rows,of,g,y,'historical')
    all_rows = np.ones(len(y),bool)
    sub = dict(all=all_rows,new_orders=~np.isin(g,og))
    result = dict(original=evaluate(orig,first,ref,y,g,fold,sub),
                  historical=evaluate(old_tree,past,champ,oy,og,of,dict(all=np.ones(len(oy),bool))))
    assert result['original']['all']['net']==29
    x = design(raw)
    t,q = ix[fold!=0],ix[fold==0]
    np.testing.assert_allclose(kernel(x,y,g,t,q),first[q],rtol=1e-10,atol=1e-10)
    out = ROOT/'artifacts/nov24_selected_kernel_veto'
    out.mkdir(exist_ok=True)
    for seed in (1021,1022):
        selected = refit(raw,y,g,seed)
        kscore = np.full(len(y),np.nan)
        for f,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,groups=g)):
            dest = out/f'{seed}_{f}.npz'
            if not dest.exists():
                score = kernel(x,y,g,t,q)
                with dest.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=t,q=q,y=y[t],p=score)
                dest.with_suffix('.tmp').replace(dest)
            with np.load(dest) as z:
                for key,val in [('t',t),('q',q),('y',y[t])]:
                    np.testing.assert_array_equal(z[key],val)
                kscore[q] = z['p']
        assert np.isfinite(kscore).all()
        result[str(seed)] = {key:evaluate(a,b,ref,y,g,fold,sub) for key,a,b in
                 [('both_refit',selected,kscore),('tree_refit',selected,first),('kernel_refit',orig,kscore)]}
    for key,r in result.items():
        r = r if 'all' in r else r['both_refit']
        print(key,r['all']['corrected'],r['all']['harmed'],r.get('new_orders',{}).get('net'))
    (ROOT/'reports/nov23_matched_learning_curve/selected_kernel_veto.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
