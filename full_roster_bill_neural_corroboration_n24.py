"""Count actual overlap and learn when independent billing and interaction models can overturn a decision."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities


def load_scores(path,raw,y,fold,keys):
    with np.load(path) as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        return np.column_stack([z[k] for k in keys])


def run():
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    _,_,og,*_ = aligned()
    unseen = ~np.isin(g,og)
    art = ROOT/'artifacts'
    with np.load(art/'nov24_remaining_cases/after_archive.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        shared = z['missed']
    inter = art/'nov24_field_interactions'
    fields = load_scores(inter/'oof.npz',raw,y,fold,('additive','pair','triple'))
    seeds = load_scores(inter/'third_order_seeds.npz',raw,y,fold,['scores'])
    booking = load_scores(art/'nov24_booking_transport/oof.npz',raw,y,fold,('individual','independent','transport'))
    np.testing.assert_array_equal(fields[:,-1],seeds[:,0])
    bills = np.full((len(y),4),np.nan)
    for f in np.unique(fold):
        query = np.flatnonzero(fold==f)
        train = np.flatnonzero(fold!=f)
        with np.load(art/f'nov24_bill_amount_frequency/full_{f}.npz') as z:
            for k,v in [('q',query),('t',train),('truth',y[train])]:
                np.testing.assert_array_equal(z[k],v)
            bills[query] = z['p']
    assert np.isfinite(bills).all()
    all_scores = np.column_stack([fields,booking,seeds[:,1:],bills])
    hit = (all_scores>=.5)==y[:,None]
    rescued = shared & hit.any(1)
    old_union = shared & hit[:,:10].any(1)
    bill_union = shared & hit[:,10:].any(1)
    report = dict(shared=int(shared.sum()), six_model_union=int((shared & hit[:,:6].any(1)).sum()),
                  interaction_seed_union=int((shared & ((seeds>=.5)==y[:,None]).any(1)).sum()),
                  six_and_seeds=int(old_union.sum()), bill_union=int(bill_union.sum()),
                  intersection=int((old_union & bill_union).sum()), total_union=int(rescued.sum()),
                  whole_population_base_errors_with_a_correct_model=int(((ref!=y) & hit.any(1)).sum()),
                  whole_population_base_correct_with_a_wrong_model=int(((ref==y) & (~hit).any(1)).sum()))
    dissent = (seeds>=.5)!=ref[:,None]
    bill_dissent = (bills[:,2]>=.5)!=ref
    booking_dissent = (booking[:,-1]>=.5)!=ref
    policies = {f'billing_and_at_least_{k}_interaction_seeds': bill_dissent & (dissent.sum(1)>=k)
                for k in (1,3,5)}
    signed = np.where(ref,-1,1)[:,None]
    def strength(p):
        return signed*logit(np.clip(p,.002,.998))
    basic = np.column_stack([ref,strength(gate[:,None]),((full>=.5)!=ref[:,None]).mean(1)])
    neural = np.column_stack([basic,strength(fields),strength(booking),strength(seeds).mean(1),
                              strength(seeds).std(1),dissent.mean(1)])
    bill_strength = strength(bills)
    both = np.column_stack([neural,bill_strength,bill_strength[:,2]-bill_strength[:,3],
                           bill_dissent*dissent.mean(1), bill_dissent*booking_dissent])
    active = dissent.any(1) | bill_dissent | booking_dissent | ((fields>=.5)!=ref[:,None]).any(1)
    for name,context in [('baseline_only',basic),('neural_and_booking',neural),('joint',both)]:
        changes = np.zeros(len(y),bool)
        for f in np.unique(fold):
            outer = np.flatnonzero(fold==f)
            for fit,held in GroupKFold(4,shuffle=True,random_state=631).split(outer,groups=g[outer]):
                t,q = outer[fit],outer[held]
                t,q = t[active[t]],q[active[q]]
                assert not np.isin(g[t],g[q]).any()
                model = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500))
                model.fit(context[t],ref[t]!=y[t])
                changes[q] = model.predict_proba(context[q])[:,1]>=.5
        policies['calibrated_'+name] = changes
    for name,changes in policies.items():
        pred = ref ^ changes
        report[name] = dict(full=stats(pred,ref,y,g,fold),
            unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
            fixed_among_199=int((shared & (pred==y)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/bill_neural_corroboration.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v if isinstance(v,int) else [v['full']['corrected'],v['full']['harmed'],v['unseen']['net'],v['fixed_among_199']]
           for k,v in report.items()})


if __name__=='__main__':
    with threadpool_limits(2):
        run()
