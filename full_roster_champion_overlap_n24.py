"""Separate errors that persist across validation designs from changes of reference model."""
import json

import numpy as np
import pandas as pd

from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference


def aligned():
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    gate = latest_reference(y,g,fold,names,half,full)
    with np.load(ROOT/'artifacts/v96_oof_completion/oof.npz') as archived:
        rows,oldfold,v96 = archived['dev_ids'],archived['fold_id'],archived['prob']
        np.testing.assert_array_equal(raw.PassengerId.iloc[rows].to_numpy(str),archived['passenger_ids'])
    with np.load(ROOT/'artifacts/nov23_incumbent_budget_transfer/oof.npz') as archived:
        for key,v in [('row',rows),('fold',oldfold),('y',y[rows])]:
            np.testing.assert_array_equal(archived[key],v)
        champion,cold = archived['v131'],archived['unseen']
    expected_cold=np.zeros(len(rows),bool)
    for h in np.unique(oldfold):
        q=oldfold==h
        expected_cold[q]=~np.isin(g[rows[q]],g[rows[~q]])
    np.testing.assert_array_equal(cold,expected_cold)
    replay = pd.read_csv(ROOT/'artifacts/nov21_case_checks/v136_final_oof.csv')
    for key,v in [('row',rows),('fold',oldfold),('true',y[rows]),('v131',champion)]:
        np.testing.assert_array_equal(replay[key].to_numpy(),v)
    old = [v96]
    selected = ['mitra96','v44_without_name','tabicl']
    for key in selected[1:]:
        with np.load(ROOT/f'artifacts/nov23_full_roster_booking_validation/original_{key}.npz') as z:
            for k,v in [('row',rows),('fold',oldfold),('y',y[rows])]:
                np.testing.assert_array_equal(z[k],v)
            old.append(z['p'])
    return (raw,y[rows],g[rows],rows,oldfold,fold[rows],champion,gate[rows],cold,
            names,full[rows],np.column_stack(old),full[rows][:,[names.index(k) for k in selected]])


def run():
    raw,y,g,rows,oldfold,newfold,champion,gate,cold,names,full,old,new = aligned()
    assert len(y)==6954 and (champion==y).sum()==5662
    a,b = (old[:,0]>=.5),(new[:,0]>=.5)
    def counts(mask):
        return dict(rows=int(mask.sum()),both_wrong=int((mask & (champion!=y) & (gate!=y)).sum()),
                    only_champion_wrong=int((mask & (champion!=y) & (gate==y)).sum()),
                    only_gate_wrong=int((mask & (champion==y) & (gate!=y)).sum()),
                    both_correct=int((mask & (champion==y) & (gate==y)).sum()))
    split = dict(all=np.ones(len(y),bool),no_historical_training_partner=cold,
                 disagreeing_v96=a!=b,same_v96=a==b,
                 neither_overridden=(champion==a)&(gate==b),any_override=(champion!=a)|(gate!=b),
                 shared_19_errors=((full>=.5)!=y[:,None]).all(1))
    expert = {}
    for k,name in enumerate(names):
        z = full[:,k]>=.5
        fixes = (gate!=y)&(z==y)
        harms = (gate==y)&(z!=y)
        expert[name] = dict(gate_fixes=int(fixes.sum()),champion_already_right=int((fixes&(champion==y)).sum()),
                            fixes_both=int((fixes&(champion!=y)).sum()),gate_harms=int(harms.sum()),
                            harms_both=int((harms&(champion==y)).sum()))
    replicated = {}
    for name,z,h in zip(['mitra96','v44_without_name','tabicl','equal_mean'],
                        [*old.T,old.mean(1)],[*new.T,new.mean(1)],strict=True):
        old_right,new_right = (z>=.5)==y,(h>=.5)==y
        fixed_old = (champion!=y)&old_right
        fixed_new = (gate!=y)&new_right
        harmed_old = (champion==y)&~old_right
        harmed_new = (gate==y)&~new_right
        replicated[name] = dict(fix_old=int(fixed_old.sum()),fix_new=int(fixed_new.sum()),
                fix_both=int((fixed_old&fixed_new).sum()),harm_old=int(harmed_old.sum()),
                harm_new=int(harmed_new.sum()),harm_both=int((harmed_old&harmed_new).sum()))
    report=dict(overlap={k:counts(v) for k,v in split.items()},expert=expert,matched_models=replicated)
    path = ROOT/'reports/nov23_matched_learning_curve/champion_overlap.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['overlap'],report['matched_models'])


if __name__=='__main__':
    run()
