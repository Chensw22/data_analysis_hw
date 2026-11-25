"""Measure held predictions against the frozen archive in its dedicated Python environment."""
import json
import numpy as np
from full_roster_booking_validation_n23 import data, ROOT
from distinct_challenger_error_reach_n26 import population
from arbitration_against_scored_nine_n26 import champions

OUT = ROOT / 'artifacts/nov26_booking_block_teacher'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/booking_block_teacher.json'


def counts(mask, before, after, truth):
    right = truth == after
    wrong = truth != before
    return dict(rows=int(mask.sum()), errors=int((mask & wrong).sum()),
                fixed=int((mask & wrong & right).sum()), harmed=int((mask & ~wrong & ~right).sum()))


def run():
    raw,y,g,fold = data()
    result = {}
    for k in range(8):
        t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(OUT / f'fold{k}.npz') as z:
            for key,expected in [('t',t),('q',q),('train_y',y[t])]:
                np.testing.assert_array_equal(z[key],expected)
            for key in z.files:
                if key not in ('t','q','train_y'):
                    result.setdefault(key,np.full(len(y),np.nan))[q] = z[key]
    assert all(np.isfinite(v).all() for v in result.values())
    _,truth,groups,f,parent,same,votes,_,_,_,solo,paid = population()
    for a,b in [(y,truth),(g,groups),(fold,f)]:
        np.testing.assert_array_equal(a,b)
    baseline,known,_,_ = champions()
    awake = raw.CryoSleep.eq(False).to_numpy() & paid
    sleeping = raw.CryoSleep.eq(True).to_numpy() & ~paid
    all_rows = np.ones(len(y),bool)
    scopes = dict(old_consensus=same, unanimous_127=same & (votes==0), lone=solo & same,
                  awake_paid_old_negative=same & awake & ~parent, sleeping_old_positive=same & sleeping & parent)
    report = dict(protocol='Eight order-disjoint outer folds; no test/outer-query covariates used in fitting. Three teachers refit per inner split using only the remaining outer-training orders. Identical seed and label-head inference for all student variants; blended snapshots at epochs 18,23,28.',
                  comparisons={}, students={})
    for name,values in result.items():
        pred = values >= .5
        report['students'][name] = dict(correct=int((pred==y).sum()), accuracy=float((pred==y).mean()),
            per_fold=[int((pred[fold==k]==y[fold==k]).sum()) for k in range(8)],
            champion_known=counts(known, baseline, pred, y),
            **{tag:counts(scope, parent,pred,y) for tag,scope in scopes.items()})
    for a,b in [('swap_labels','scratch'),('swap_mean','swap_labels'),('swap_individual','swap_mean'),
                ('swap_individual','scratch')]:
        p0,p1 = result[b]>=.5,result[a]>=.5
        report['comparisons'][a+'_vs_'+b] = {m:counts(scope,p0,p1,y) for m,scope in
            dict(all=all_rows,**scopes, originally_unanimous_wrong=same & (parent!=y)).items()}
        report['comparisons'][a+'_vs_'+b]['fold_net'] = [int((p1[fold==k]==y[fold==k]).sum()-(p0[fold==k]==y[fold==k]).sum()) for k in range(8)]
    # Deployable broad switch: consult all three teachers, with and without the independently distilled student.
    teacher_hard = np.column_stack([result[k]>=.5 for k in ('extra_trees','hist_boost','additive_logistic')])
    united = (teacher_hard==teacher_hard[:,:1]).all(1)
    agree_student = teacher_hard[:,0]==(result['swap_individual']>=.5)
    support = {'teacher_unanimity':united, 'teacher_unanimity_and_student':united & agree_student}
    report['switch_champion'] = {}
    for key,consent in support.items():
        called = np.where(consent,teacher_hard[:,0],baseline)
        report['switch_champion'][key] = {tag:counts(mask,baseline,called,y) for tag,mask in
            dict(champion_known=known, **scopes).items()}
        report['switch_champion'][key]['nets_by_fold'] = [int(((called==y)&known&(f==k)).sum()-
            ((baseline==y)&known&(f==k)).sum()) for k in range(8)]
    # Did separate teachers acquire any new evidence beyond the cheaper mean-target student?
    own_opposition = (result['swap_individual']>=.5)!=parent
    mean_support = (result['swap_mean']>=.5)==parent
    novel = own_opposition & mean_support
    report['separate_only'] = {tag:counts(mask, parent,np.where(novel,~parent,parent),y)
                               for tag,mask in scopes.items()}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['comparisons']['swap_individual_vs_scratch']['all'],flush=True)


if __name__ == '__main__':
    run()
