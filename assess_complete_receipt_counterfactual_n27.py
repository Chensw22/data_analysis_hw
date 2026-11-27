"""Evaluate facility deletion on complete bills against the original, all deletions, and a masked tree."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from complete_receipt_counterfactual_n27 import STORE, REPORT, splits, FEES, ART, inputs, bank_for, source_folds, data_dir
from exaone_validation_data import compact_features, CATEGORICAL
from booking_mix_confidence_transfer_n26 import partition
from repeated_archive_misses_n26 import cohort, count

def selectors(protocol):
    frame, y, g, f, _ = inputs()
    x = pd.get_dummies(compact_features(frame), columns=CATEGORICAL).astype(float)
    x[FEES] = np.log1p(x[FEES])
    values = x[FEES].to_numpy()
    complete = np.isfinite(values).all(1)
    surprise = np.full(values.shape, -1.)
    reserve = partition(g, 'complete-receipt-amount-calibration', 5) == 0
    for k, t, q in splits(frame, y, g, source_folds(g, f, protocol)):
        for j, name in enumerate(FEES):
            eligible = complete[t] & (values[t, j] > 0)
            fit = t[eligible & ~reserve[t]]
            hold = t[eligible & reserve[t]]
            assert not np.isin(g[fit], g[hold]).any()
            rest = x.drop(columns=name)
            model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, l2_regularization=2, random_state=7253)
            model.fit(rest.iloc[fit], values[fit, j])
            residuals = np.sort(abs(model.predict(rest.iloc[hold]) - values[hold, j]))
            apply = q[values[q, j] > 0]
            discrepancy = abs(model.predict(rest.iloc[apply]) - values[apply, j])
            surprise[apply, j] = np.searchsorted(residuals, discrepancy, side='right') / len(hold)
    STORE.mkdir(exist_ok=True)
    name = 'selection.npz' if protocol=='random' else 'selection_ordered.npz'
    np.savez_compressed(STORE/name, ids=frame.PassengerId.to_numpy(str), surprise=surprise)


def select(curve, signal, paid):
    positive_max = signal == np.max(signal, axis=1, keepdims=True)
    weight = positive_max / positive_max.sum(1, keepdims=True).clip(1)
    return np.where(paid, np.sum(curve[:, 1:]*weight, axis=1), curve[:, 0])


def assess(replicas, protocol):
    from full_roster_bill_subset_route_n24 import BILLS
    assert set(BILLS) == set(FEES)
    frame, y, g, f, _ = inputs()
    ids = frame.PassengerId.to_numpy(str)
    n = len(y)
    base161, blind, known, fresh = cohort(frame.iloc[:n], y, f)
    values = frame[FEES].to_numpy(float)
    complete = np.isfinite(values).all(1)
    paid = complete & (np.nan_to_num(values) > 0).any(1)
    values = np.nan_to_num(values)
    with np.load(STORE/('selection.npz' if protocol=='random' else 'selection_ordered.npz')) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        surprise = z['surprise']
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'], ids)
        anchor = z['parent']
        outside = np.column_stack([z['dpt'], np.median(z['tabnet'],axis=1)])
    best = pd.read_csv(ART/'proposal_n27_directional_foundation/submission.csv')
    np.testing.assert_array_equal(best.PassengerId, ids[n:])
    np.testing.assert_array_equal(best.Transported, anchor[n:])
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'], ids)
        np.testing.assert_array_equal(z['names'][-3:], ['Mitra','TabICL fare','Nori fare'])
        panel = z['scores']
        confidence = np.median(panel[:, -3:], axis=1)
        wide = np.column_stack([np.median(panel[:,:18],axis=1), panel[:,-3:], confidence, outside])
        unanimous = ((wide>=.5)==anchor[:,None]).all(1)
    with np.load(ART/'nov26_latent_bill_visibility/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'], ids)
        treecurve = z['curves'][:, [31, *(31^(1<<BILLS.index(name)) for name in FEES)]]
    curves = {}
    for rep in range(replicas):
        curve = np.full((len(ids), 6), np.nan)
        for k, t, q in splits(frame,y,g,source_folds(g, f, protocol)):
            bank = bank_for(t,g,k,rep)
            for j, name in enumerate(['original',*FEES]):
                with np.load(data_dir(rep,protocol,k)/f'{k}_{name}.npz') as z:
                    for field,expected in [('bank',bank),('q',q),('ids',ids[q]),('labels',y[bank])]:
                        np.testing.assert_array_equal(z[field],expected)
                    curve[q,j] = z['probability']
        assert np.array_equal(np.isfinite(curve).all(1),complete)
        curves[f'bank{rep}'] = curve
    curves['banks_mean'] = np.mean(list(curves.values()),axis=0)
    curves['tree'] = treecurve
    estimated = {}
    for arch, curve in curves.items():
        estimated[arch] = dict(original=curve[:,0], average=curve[:,1:].mean(1),
                               largest=select(curve,values,paid), surprise=select(curve,surprise,paid))
    odds = lambda a: logit(a.clip(.005,.995))
    native = np.where(anchor, 1., -1.) * abs(odds(confidence))
    opposite = np.where(anchor, -1., 1.)
    def overturn(change):
        return change*opposite > np.abs(native)
    baseline_wrong = anchor[:n]!=y
    population = dict(complete=known&complete[:n],paid=known&paid[:n],fresh=known&fresh&paid[:n],
                      zero=known&complete[:n]&~paid[:n],seven_agree=known&paid[:n]&unanimous[:n],common=blind&paid[:n]&(anchor[:n]==base161))
    primary_folds = source_folds(g, f, protocol)
    def summary(edit):
        delta = edit[:n].astype(int)*np.where(baseline_wrong,1,-1)
        return dict(scopes={s:count(edit[:n],mask,baseline_wrong) for s,mask in population.items()},
                    fold_net=[int(delta[known&(primary_folds==k)].sum()) for k in range(8)],test=int(edit[n:].sum()))
    actions = {}
    native_accuracy = {}
    for arch, ps in estimated.items():
        native_accuracy[arch] = {}
        for key,p in ps.items():
            side = p >= .5
            acc = {}
            for key2, mask in dict(complete=complete[:n],paid=paid[:n],zero=complete[:n]&~paid[:n]).items():
                correct = side[:n]==y
                original_ok = (ps['original'][:n]>=.5)==y
                acc[key2]=dict(rows=int(mask.sum()),correct=int(correct[mask].sum()),accuracy=float(correct[mask].mean()),
                               fixed=int((mask&correct&~original_ok).sum()),harmed=int((mask&~correct&original_ok).sum()))
            native_accuracy[arch][key]=acc
            actions[f'{arch}_{key}_direct'] = paid & (side!=anchor)
            if key not in ('largest','surprise'):
                continue
            against = []
            for ctrl in ('original','average'):
                diff = odds(p)-odds(ps[ctrl])
                against.append(paid & overturn(diff))
            actions[f'{arch}_{key}_paired'] = against[0]&against[1]
            actions[f'{arch}_{key}_restore'] = against[0]&against[1]&(side!=anchor)
            # Exploratory after random fold 0: distinguish a surprising bill from an unreliable
            # one by reinforcing its original contribution. The contiguous outer refit is separate.
            reinforce = odds(ps['original'])-odds(p)
            actions[f'{arch}_{key}_reinforce'] = paid & overturn(reinforce) & ((ps['original']>=.5)!=anchor)

    for key in ('largest','surprise'):
        replicated = np.logical_and.reduce([actions[f'bank{r}_{key}_restore'] for r in range(replicas)])
        actions[key+'_replicated'] = replicated
        actions[key+'_tree_confirmed'] = replicated & actions['tree_'+key+'_direct']
        actions[key+'_tree_paired'] = replicated & actions['tree_'+key+'_restore']
        actions[key+'_reinforce_tree'] = actions[f'tree_{key}_reinforce'] & np.logical_and.reduce(
            [actions[f'bank{r}_{key}_reinforce'] for r in range(replicas)])
    fits = {}
    # On the contiguous refit, calibrate only from original random-fold scores on other
    # orders in the query's original outer fold. Both the historical anchor and the source
    # scores then exclude the query's labels. Apply these same coefficients to the newly
    # trained contiguous-order TabICL. Test scores must be identical under either evaluation.
    training_scores = estimated['banks_mean']
    if protocol=='ordered':
        with np.load(STORE/f'aligned_random_{replicas}.npz') as cached, np.load(STORE/'selection.npz') as z:
            np.testing.assert_array_equal(cached['ids'],ids)
            np.testing.assert_array_equal(z['ids'],ids)
            original_curve = cached['curve_banks_mean']
            training_scores = dict(original=original_curve[:,0],average=original_curve[:,1:].mean(1),
                                   largest=select(original_curve,values,paid),surprise=select(original_curve,z['surprise'],paid))
    weights = {}
    for mode in ('random','ordered'):
        bins = partition(g[:n],'complete-receipt-translation',4)
        if mode=='ordered':
            for k in range(8):
                for j,v in enumerate(np.array_split(np.unique(g[:n][f==k]),4)):
                    bins[(f==k)&np.isin(g[:n],v)] = j
        for key in ('largest','surprise','average'):
            ps = estimated['banks_mean']
            controls = (ps['original'],) if key=='average' else (ps['original'], ps['average'])
            delta = np.nan_to_num(odds(ps[key])-np.mean([odds(p) for p in controls],axis=0))
            train_controls = (training_scores['original'],) if key=='average' else (training_scores['original'], training_scores['average'])
            fit_delta = np.nan_to_num(odds(training_scores[key])-np.mean([odds(p) for p in train_controls],axis=0))
            score = np.zeros(len(ids))
            production = []
            coeffs = []
            for k in range(8):
                for j in [0,1,2,3,None]:
                    t = np.flatnonzero((f==k)&known&paid[:n]&((bins!=j) if j is not None else True))
                    q = np.flatnonzero((f==k)&(bins==j)) if j is not None else np.arange(n,len(ids))
                    assert not np.isin(g[t],g[q]).any()
                    def loss(beta):
                        z = native[t]+beta*fit_delta[t]
                        return np.sum(np.logaddexp(0,z)-y[t]*z)+2*beta**2
                    fit = minimize_scalar(loss,bounds=(0.,2.),method='bounded')
                    assert fit.success
                    beta = fit.x
                    coeffs.append(beta)
                    if j is None:
                        production.append(beta)
                        score[q] += beta*delta[q]/8
                    else:
                        score[q] = beta*delta[q]
            weights[f'{key}_{mode}'] = np.asarray(production)
            changed = paid&overturn(score)
            actions[f'{key}_{mode}_calibrated'] = changed
            if key!='average':
                actions[f'{key}_{mode}_confirmed'] = changed & actions[f'{key}_tree_confirmed']
            fits[f'{key}_{mode}'] = dict(mean=float(np.mean(coeffs)),median=float(np.median(coeffs)))
    oracle = {}
    for arch,curve in curves.items():
        original_ok = (curve[:n,0]>=.5)==y
        hidden_ok = (curve[:n,1:]>=.5)==y[:,None]
        by_scope = {}
        for name,domain in population.items():
            miss = domain & baseline_wrong
            by_scope[name] = dict(reference_errors=int(miss.sum()), original_correct=int((miss&original_ok).sum()),
                                  rescued_by_any_delete=int((miss&~original_ok&hidden_ok.any(1)).sum()),
                                  all_six_miss=int((miss&~original_ok&~hidden_ok.any(1)).sum()),
                                  all_five_correct=int((miss&~original_ok&hidden_ok.all(1)).sum()))
        oracle[arch] = by_scope
    out = dict(oracle=oracle,anchor='proposal_n27_directional_foundation',eligible_train=int(paid[:n].sum()),eligible_test=int(paid[n:].sum()),
               complete_train=int(complete[:n].sum()),complete_test=int(complete[n:].sum()),
               original_common_errors=int((population['common']&baseline_wrong).sum()),native=native_accuracy,
               fits=fits,results={k:summary(a) for k,a in actions.items()})
    path = REPORT if replicas==2 else REPORT.with_name(REPORT.stem+'_one_bank.json')
    if protocol=='ordered': path = REPORT.with_name(REPORT.stem+('_ordered_refit.json' if replicas==2 else '_ordered_one_bank.json'))
    path.write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(STORE/f'aligned_{protocol}_{replicas}.npz',ids=ids,fold=primary_folds,y=y,anchor=anchor,native=native,**{'weights_'+k:p for k,p in weights.items()},**actions,**{'curve_'+k:p for k,p in curves.items()})
    for k,a in actions.items():
        if k.startswith(('largest','surprise','average')): print(k,out['results'][k],flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',choices=['selectors','one','two'])
    ap.add_argument('--protocol',choices=['random','ordered'],default='random')
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        if args.mode=='selectors': selectors(args.protocol)
        else: assess(1 if args.mode=='one' else 2, args.protocol)
