"""Attribute errors on identical edit budgets, preserving each held fold's coverage."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import conflict_pair_trigger_n26 as experiment

REPORT = experiment.ROOT/'reports/nov23_matched_learning_curve/fixed_pair_budget.json'


def audit(mass,rank,individual,errors,pairs,fold):
    pair_fold = fold[pairs[:,0]]
    np.testing.assert_array_equal(pair_fold,fold[pairs[:,1]])
    count = errors.sum(1)
    member = np.where(rank>=.5,0,1)
    pred_correct = errors[np.arange(len(pairs)),member]
    expected = mass[:,2]+mass[:,1]*np.maximum(rank,1-rank)
    picked = expected>.5
    def describe(selected,correct):
        select = count[selected]
        good = correct[selected]
        zero = int((select==0).sum())
        misrank = int(((select==1)&~good).sum())
        assert int((~good).sum())==zero+misrank
        return dict(selected=len(select),zero=zero,one=int((select==1).sum()),two=int((select==2).sum()),
                    misrank=misrank,fixes=int(good.sum()),net=int(2*good.sum()-len(select)))
    out = describe(picked,pred_correct)
    out['by_fold'] = [describe(picked&(pair_fold==k),pred_correct) for k in np.unique(pair_fold)]
    selector_scores = dict(edited_member_risk=expected,any_error=mass[:,1:].sum(1),
                           max_individual_risk=individual.max(1))
    actual = dict(pair_rank=pred_correct,
                  individual_rank=errors[np.arange(len(pairs)),np.argmax(individual,axis=1)])
    out['same_budget'] = {}
    for name,score in selector_scores.items():
        selected = np.zeros(len(pairs),bool)
        for k in np.unique(pair_fold):
            ids = np.flatnonzero(pair_fold==k)
            n = int(picked[ids].sum())
            preferred = ids[np.argsort(-score[ids],kind='stable')[:n]]
            selected[preferred] = True
        if name=='edited_member_risk':
            np.testing.assert_array_equal(selected,picked)
        out['same_budget'][name] = {key:describe(selected,hit) for key,hit in actual.items()}
    chosen = np.flatnonzero(picked)
    out['mean_forecast_success_on_chosen'] = float(expected[chosen].mean())
    out['mean_forecast_any_error_on_chosen'] = float((1-mass[chosen,0]).mean())
    one = picked&(count==1)
    out['mean_forecast_rank_success_if_one_on_chosen'] = float(np.maximum(rank[one],1-rank[one]).mean())
    return out


def run():
    _,_,_,fold = experiment.data()
    saved = json.loads(experiment.REPORT.read_text())
    captured=[]
    original_measure=experiment.measure
    def record(mass,rank,base,errors,pairs,keep):
        result = original_measure(mass,rank,base,errors,pairs,keep)
        captured.append(audit(mass[keep],rank[keep],base[keep],errors[keep],pairs[keep],fold))
        return result
    # Use the exact predictions, comparison scopes and protocol of the frozen experiment.
    # Redirect its output to a temporary directory so the existing report stays immutable.
    with TemporaryDirectory() as tmp:
        dest = Path(tmp)/'rerun.json'
        with patch.object(experiment,'measure',record),patch.object(experiment,'REPORT',dest):
            experiment.run()
        assert json.loads(dest.read_text())==saved
    report={}
    it=iter(captured)
    for setting in ('nearest','confidence_gap'):
        report[setting]={}
        for scope in ('agree_with_anchor','all_eleven_agree'):
            result=next(it)
            origin = saved[setting][scope]['both_learned']
            assert (result['selected'],result['fixes'],result['net'])==(origin['selected'],origin['fixed'],origin['net'])
            report[setting][scope]=result
    assert next(it,None) is None
    report['scope']='With at most one edit per pair, every harmful edit is either a pair with no error or a wrong side chosen on an exactly-one-error pair. Selection diagnostics use the original learned threshold. Alternative selectors pick the same number of pairs within each held fold as the original threshold, using only model scores. Ordering losses compare identical selected sets.'
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
