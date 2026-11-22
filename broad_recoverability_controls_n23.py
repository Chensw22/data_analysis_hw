"""Measure recoverability against matched random and visibly structured error assignments."""
import numpy as np
from scipy.special import logit, expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_refit_dissent_n23 import evidence, features
from broad_weak_disagreement_n23 import OUT, REPORT, SEEDS, archive
from relational_error_corrector import write


def assign(error, base, region, marker, ids, rng, structured):
    risk = np.minimum(base[ids], 1-base[ids])
    # Match the number of baseline errors in every block exactly, including the
    # direction of the decision, persistent dissent and absolute confidence.
    cell = np.floor(10*risk).astype(int)*4 + region[ids].astype(int)*2 + (base[ids] >= .5)
    dest = np.zeros(len(ids), bool)
    ranking = rng.normal(size=len(ids))
    if structured:
        ranking += marker[ids]
    for val in np.unique(cell):
        c = np.flatnonzero(cell == val)
        n = error[ids[c]].sum()
        dest[c[np.argsort(ranking[c])[:n]]] = True
        assert dest[c].sum() == n
    return dest


def risk_predictions(z, base, t, q, err):
    prob = np.minimum(base, 1-base).clip(.001,.4999)
    # Correct for a global intercept, slope and positive/negative decision bias.
    x = np.column_stack([logit(prob), base >= .5])
    lr = LogisticRegression(C=1.).fit(x[t],err)
    prior = lr.predict_proba(x)[:,1]
    # Learn excess error risk over the calibrated prior; shrink through leaf
    # size and learning rate. Query residuals are never used for training.
    learner = HistGradientBoostingRegressor(max_iter=75, learning_rate=.045, max_leaf_nodes=7,
                                              min_samples_leaf=40, l2_regularization=10, early_stopping=False)
    learner.fit(z[t], err.astype(float)-prior[t])
    score = np.clip(prior[q]+learner.predict(z[q]), .0001, .9999)
    return prior[q], score


def diagnostic(error, scores, region, archive_set):
    result = {}
    for name, scope in (("all", np.ones(len(error),bool)), ("refit_region", region), ("archive_pattern", archive_set)):
        truth, risk = error[scope], scores[scope]
        flip = risk >= .5
        gain = np.where(flip, 2*truth.astype(int)-1, 0)
        order = np.argsort(-risk,kind="stable")
        result[name] = dict(rows=int(scope.sum()),base_errors=int(truth.sum()),
                            auc=float(roc_auc_score(truth, risk)), fixed=int((gain>0).sum()), harmed=int((gain<0).sum()),
                            precision_top_10pct=float(truth[order[:max(1,round(.1*len(order)))]].mean()))
    return result


def run():
    row, y, frame, g, _ = load()
    old, old_y, _, archive_set, cohorts, _, _ = archive()
    np.testing.assert_array_equal(row, old)
    np.testing.assert_array_equal(y, old_y)
    raw = features(frame)[0]
    # The positive control uses a known direction of association among observed
    # spending values. Overall error frequencies and the raw inputs stay fixed.
    marker = np.log1p(frame.Spa.fillna(0)) - np.log1p(frame.FoodCourt.fillna(0))
    marker = (marker / 2).to_numpy()
    records, hist = {}, {}
    for name in [*map(str, SEEDS), "v131"]:
        with np.load(OUT/f"decisions_{name}.npz") as z:
            np.testing.assert_array_equal(row,z["row"])
            base, fold, ref, overrides = (z[k] for k in ("base_probability","fold","ref","overrides"))
            defined_region = z["region"]
        err = (base >= .5) != y
        scores = {(source, fit):np.full(len(y),np.nan) for source in ("actual","random","structured")
                  for fit in ("confidence","fields_and_opinions")}
        outcome = {s:np.zeros(len(y),bool) for s,_ in scores}
        seen = np.zeros(len(y),int)
        for k in np.unique(fold):
            q = np.flatnonzero(fold==k)
            with np.load(OUT/f"refits_{name}_{k}.npz") as z:
                np.testing.assert_array_equal(q,z["q"])
                members = z["members"]
            views, region = evidence(base[q], members)
            np.testing.assert_array_equal(defined_region[q],region)
            design = np.column_stack([views["repeat"], raw[q]])
            for j,(a,b) in enumerate(GroupKFold(3, shuffle=True, random_state=667).split(q,groups=g[q])):
                assert not np.intersect1d(g[q[a]],g[q[b]]).size
                seen[q[b]] += 1
                for source in outcome:
                    if source == "actual":
                        train, test = err[q[a]], err[q[b]]
                    else:
                        # Separate assignments within calibration and evaluation.
                        rng = np.random.default_rng(211 + 7*int(k) + j)
                        train = assign(err[q],base[q],region,marker[q],a,rng,source=="structured")
                        test = assign(err[q],base[q],region,marker[q],b,rng,source=="structured")
                    outcome[source][q[b]] = test
                    prior, learned = risk_predictions(design,base[q],a,b,train)
                    scores[source,"confidence"][q[b]] = prior
                    scores[source,"fields_and_opinions"][q[b]] = learned
        assert (seen==1).all()
        records[name] = {}
        for (source,fit), risk in scores.items():
            assert np.isfinite(risk).all()
            records[name][source+"_"+fit] = diagnostic(outcome[source],risk,defined_region,archive_set)
            if source=="actual":
                final = np.where(overrides,ref,np.logical_xor(base>=.5,risk>=.5))
                hit = final==y
                gain = hit.astype(int)-(ref==y)
                records[name][source+"_"+fit]["deployed"] = dict(
                    accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                    gain_by_fold=[int(gain[fold==k].sum()) for k in np.unique(fold)],
                    fixes={k:int(((gain>0)&m).sum()) for k,m in cohorts.items()},
                    harms={k:int(((gain<0)&m).sum()) for k,m in cohorts.items()})
                if name!="v131":
                    hist.setdefault(fit,[]).append((hit,gain))
        write(REPORT/"recoverability.json",dict(results=records))
        print(name,records[name]["actual_fields_and_opinions"]["deployed"],flush=True)
    stable = {}
    for fit, trials in hist.items():
        hit = np.logical_and.reduce([h for h,_ in trials])
        stable[fit] = {k:int((hit&m).sum()) for k,m in cohorts.items()}
        stable[fit]["paired_ci_pp"] = interval(np.mean([v for _,v in trials],0),g)
    write(REPORT/"recoverability.json",dict(results=records,stable=stable))


if __name__ == "__main__":
    with threadpool_limits(2):
        run()
