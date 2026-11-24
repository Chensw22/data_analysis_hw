"""Count paired error coverage and test agreement across kernel and spending model families."""
import json

import numpy as np

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats


def from_cache(tag, rows, fold, y, g):
    names = ['global_metric', 'error_partition', 'service_independent', 'service_joint']
    score = np.full((len(rows),len(names)),np.nan)
    for f in np.unique(fold):
        held = fold == f
        q = rows[held]
        t = rows[~held & ~np.isin(g[rows],g[q])]
        for folder, key, source, dest in [
            ('recursive_metric','prob',[3],[0]), ('error_partition_kernel','prob',[2],[1]),
            ('service_order_mixture','p',[0,2],[2,3])]:
            path = ROOT/f'artifacts/nov24_{folder}/{tag}_{f}.npz'
            with np.load(path) as z:
                for k,v in [('train',t),('query',q),('truth',y[t])]:
                    np.testing.assert_array_equal(z[k],v)
                score[np.ix_(held,dest)] = z[key][:,source]
    assert np.isfinite(score).all()
    return names,score


def choices(score,base,ref):
    # Average probabilities inside each family before comparing the two families.
    kernel,bill = score[:,:2].mean(1),score[:,2:].mean(1)
    agree = (kernel>=.5)==(bill>=.5)
    free = (base>=.5)==ref
    stronger = np.minimum(np.abs(kernel-.5),np.abs(bill-.5))>np.abs(base-.5)
    return dict(both=np.where(free & agree,kernel>=.5,ref),
                both_more_confident=np.where(free & agree & stronger,kernel>=.5,ref),
                quarter=np.where(free,.75*base+.125*(kernel+bill)>=.5,ref))


def run():
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    labels,a = from_cache('historical',rows,ofold,y,g)
    _,b = from_cache('full',np.arange(len(y)),fold,y,g)
    old_hit = (a>=.5)==oy[:,None]
    hit = (b>=.5)==y[:,None]
    shared = ((full>=.5)!=y[:,None]).all(1)
    hard = shared[rows] & (champ!=oy)
    new_orders = ~np.isin(g,og)
    eligible = ref!=y
    report = {}
    for label,idx in [('kernels',[0,1]),('bill_models',[2,3]),('all_four',list(range(4)))]:
        success = hit[:,idx].any(1)
        repeated = (old_hit[:,idx] & hit[rows][:,idx]).any(1)
        report[label] = dict(any_correct_on_baseline_errors=int((eligible & success).sum()),
            any_wrong_on_baseline_correct=int((~eligible & ~hit[:,idx].all(1)).sum()),
            any_correct_on_common=int((shared & success).sum()),
            same_method_twice_on_persistent=int((hard & repeated).sum()))
    by_model = {}
    for j,label in enumerate(labels):
        by_model[label] = dict(common_fixed=int(hit[shared,j].sum()),
            common_only_this_method=int((shared & hit[:,j] & (hit.sum(1)==1)).sum()),
            common_by_correct_direction=dict(transported=int((shared & y & hit[:,j]).sum()),
                                             not_transported=int((shared & ~y & hit[:,j]).sum())))
    report['by_model'] = by_model
    original = choices(a,old[:,0],champ)
    current = choices(b,full[:,names.index('mitra96')],ref)
    report['rules'] = {}
    for k,p in current.items():
        report['rules'][k] = dict(full=stats(p,ref,y,g,fold),historical=stats(original[k],champ,oy,og,ofold),
             new_orders=stats(p[new_orders],ref[new_orders],y[new_orders],g[new_orders],fold[new_orders]),
             common_fixed=int((shared & (p==y)).sum()),persistent_fixed=int((hard & (original[k]==oy) & (p[rows]==oy)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/kernel_repair_overlap.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k] for k in ('kernels','bill_models','all_four')})
    print([(k,v['historical']['net'],v['full']['corrected'],v['full']['harmed'],v['common_fixed']) for k,v in report['rules'].items()])


if __name__=='__main__':
    run()
