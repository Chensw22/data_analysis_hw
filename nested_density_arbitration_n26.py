"""Fit both disagreeing experts without either of the paired validation folds."""
import argparse
import json
import os
import sys

import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from full_roster_booking_validation_n23 import ROOT, data
import ordinal_factor_density_n26 as density
import original_tabstar_probe_n26 as tabstar

OUT = ROOT / "artifacts/nov26_nested_density_arbitration"
REPORT = ROOT / "reports/nov23_matched_learning_curve/nested_density_arbitration.json"


def fit_pair(k):
    import torch
    torch.set_num_threads(2)
    raw, y, groups, fold = data()
    partner = k + 1
    held = np.isin(fold, [k, partner])
    t, q = np.flatnonzero(~held), np.flatnonzero(held)
    assert not np.isin(groups[t], groups[q]).any()
    path = OUT / f"pair{k}_{partner}"
    path.mkdir(parents=True, exist_ok=True)
    if not (path / "density.npz").exists():
        frame = density.features(raw, groups)
        context = frame.drop(columns=density.BILLS).copy()
        for field in density.CATEGORIES:
            context[field] = context[field].fillna("__missing__").astype(str)
        predictions = density.score_split(frame, context, y, t, q, k)
        np.savez_compressed(path / "density.npz", t=t, q=q, train_y=y[t],
                            passenger=raw.PassengerId.iloc[q].to_numpy(str), **predictions)
    if not (path / f"tuned{k}.npz").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["E5_SMALL_LOCAL_PATH"] = str(tabstar.ART/"e5")
        sys.path.insert(0, str(tabstar.ART/"package"))
        tabstar.run(k, finetune=True, extra_holdout=(partner,), output_dir=path)


def read_experts(raw, y, groups, expected_train, expected_query, density_path, tabstar_path):
    for path in (density_path, tabstar_path):
        with np.load(path) as z:
            np.testing.assert_array_equal(z["t"], expected_train)
            np.testing.assert_array_equal(z["q"], expected_query)
            np.testing.assert_array_equal(z["train_y"], y[expected_train])
            np.testing.assert_array_equal(z["passenger"], raw.PassengerId.iloc[expected_query])
    assert not np.isin(groups[expected_train],groups[expected_query]).any()
    with np.load(tabstar_path) as z:
        assert str(z["checkpoint"])==tabstar.SHA
        learned = z["finetuned"]
    with np.load(density_path) as z:
        return np.column_stack([learned, z["conditional_bills"], z["joint"], z["independent"], z["context_only"]])


def case_features(raw, groups):
    frame = density.features(raw, groups)
    bills = frame[density.BILLS].to_numpy(float)
    age = frame.Age.to_numpy(float)
    return np.column_stack([
        np.log1p(np.nan_to_num(bills)), (bills > 0).sum(axis=1), np.isnan(bills).sum(axis=1),
        np.log1p(np.nansum(bills,axis=1)), np.nan_to_num(age,nan=-1), np.isnan(age),
        frame.BookingSize, frame.CryoSleep.eq(True).fillna(False), frame.HomePlanet.eq("Earth").fillna(False),
        frame.HomePlanet.eq("Europa").fillna(False), frame.Deck.eq("G").fillna(False),
    ]).astype(float)


def offset_model(cal_x, query_x, calibration_labels, cal_logit, query_logit):
    """Estimate residual log odds with a fixed ridge prior; TabSTAR has unit initial weight."""
    scaler = StandardScaler().fit(cal_x)
    train = np.column_stack([np.ones(len(cal_x)), scaler.transform(cal_x).clip(-6,6)])
    query = np.column_stack([np.ones(len(query_x)), scaler.transform(query_x).clip(-6,6)])
    penalty = np.full(train.shape[1], .08)
    penalty[0] = .002
    def objective(beta):
        score = cal_logit + train @ beta
        nll = (np.logaddexp(0, score) - calibration_labels*score).mean()
        grad = train.T @ (expit(score)-calibration_labels)/len(score)
        return nll + (penalty @ beta**2)/2, grad + penalty*beta
    solution = minimize(objective, np.zeros(train.shape[1]), jac=True, method="L-BFGS-B")
    assert solution.success
    return expit(query_logit + query @ solution.x)


def design(experts, fields):
    logits = logit(experts.clip(.001,.999))
    tab, den, joint, ind, ctx = logits.T
    dry = fields[:,5]==0
    group = fields[:,10]>1
    bill_evidence = den-ctx
    gap = den-tab
    # A matched control retains TabSTAR, the transaction summaries and raw booking fields.
    control = np.column_stack([tab,fields])
    augmented = np.column_stack([control,gap,joint-tab,ind-tab,bill_evidence,gap*dry,gap*group])
    # This gate estimates density's chance of being right conditional on two opposite labels.
    # The direction is reset to density's vote, so confidence is comparable for either class.
    sign = np.where(den>=0,1,-1)
    witnesses = np.column_stack([den>=0, np.abs(den), np.abs(tab), sign*joint, sign*ind,
                                 sign*ctx, sign*bill_evidence, fields])
    return control,augmented,witnesses


def decide(cal, query, fields_cal, fields_query, y_cal):
    cal_x, aug_cal, gate_cal = design(cal, fields_cal)
    query_x, aug_query, gate_query = design(query, fields_query)
    tab, den = query[:,:2].T
    ctab, cden = cal[:,:2].T
    cdis = (ctab>=.5)!=(cden>=.5)
    dis = (tab>=.5)!=(den>=.5)
    success = (cden[cdis]>=.5)==y_cal[cdis]
    assert cdis.sum()>=40 and np.unique(success).size==2
    scaler = StandardScaler().fit(gate_cal[cdis])
    gate = LogisticRegression(C=.025,max_iter=500).fit(scaler.transform(gate_cal[cdis]),success)
    trust = gate.predict_proba(scaler.transform(gate_query))[:,1]
    margin = logit(tab.clip(.001,.999))
    calibration_margin = logit(ctab.clip(.001,.999))
    base = offset_model(cal_x,query_x,y_cal,calibration_margin,margin)
    adjusted = offset_model(aug_cal,aug_query,y_cal,calibration_margin,margin)
    picked = np.where(dis,np.where(den>=.5,trust,1-trust),tab)
    return dict(tabstar=tab,density=den,trust=trust,gate=picked,control=base,adjusted=adjusted)


def grouped_interval(delta, eligible, groups):
    _, index = np.unique(groups[eligible], return_inverse=True)
    group_size = np.bincount(index)
    changes = np.bincount(index, weights=delta[eligible])
    rng = np.random.default_rng(80612)
    bootstrap = []
    for _ in range(20):
        draw = rng.integers(len(changes),size=(50,len(changes)))
        bootstrap.extend(changes[draw].sum(axis=1)/group_size[draw].sum(axis=1))
    return [float(v) for v in np.quantile(bootstrap,[.025,.975])]


def summarize(y, folds, output, views, nine, known, fields, groups):
    from sklearn.metrics import log_loss, roc_auc_score
    tab, den = [output[n]>=.5 for n in ("tabstar","density")]
    disagree = tab!=den
    result = {}
    for n in ("tabstar","density","gate","control","adjusted"):
        p = output[n]
        result[n] = dict(accuracy=float(((p>=.5)==y).mean()),log_loss=log_loss(y,p),
                         per_fold=[float(((p[folds==k]>=.5)==y[folds==k]).mean()) for k in range(8)])
    result["gate_on_disagreement"] = dict(
        rows=int(disagree.sum()),density_correct=int(((den==y)&disagree).sum()),
        tabstar_correct=int(((tab==y)&disagree).sum()),
        density_correct_selected=int(((den==y)&disagree&(output["trust"]>=.5)).sum()),
        density_incorrect_selected=int(((den!=y)&disagree&(output["trust"]>=.5)).sum()),
        auc=roc_auc_score((den==y)[disagree],output["trust"][disagree]))
    overlap = {}
    for name,(mask,baseline) in views.items():
        common = mask & known & (nine==baseline)
        wrong = common & (baseline!=y)
        correct = common & (baseline==y)
        gate = output["gate"]>=.5
        control,adjusted = output["control"]>=.5,output["adjusted"]>=.5
        # New evidence is credited on a release change only when the matched control
        # would keep the release and the density vote favors the change.
        actions = dict(density_direct=den!=baseline,tabstar_direct=tab!=baseline,
                       gated_direct=gate!=baseline,adjusted_direct=adjusted!=baseline,
                       control_direct=control!=baseline,
                       accept_density_increment=(den!=baseline)&(tab==baseline)&(output["trust"]>=.5),
                       accept_density_increment_65=(den!=baseline)&(tab==baseline)&(output["trust"]>=.65),
                       accept_density_increment_75=(den!=baseline)&(tab==baseline)&(output["trust"]>=.75),
                       residual_change_attributable_to_density=(adjusted!=baseline)&(control==baseline)&(den!=baseline))
        record = {"wrong":int(wrong.sum()),"correct":int(correct.sum())}
        for action,flip in actions.items():
            record[action] = dict(fixed=int((flip&wrong).sum()),harmed=int((flip&correct).sum()))
        overlap[name] = record
    result["shared_release_decisions"] = overlap
    valid, baseline = views["full"]
    entire = np.ones(len(y),dtype=bool)
    common = valid & known & (baseline==nine)
    increment = common & (den!=baseline) & (tab==baseline) & (output["trust"]>=.5)
    result["paired_booking_bootstrap_95"] = {
        name: grouped_interval(new.astype(int)-old.astype(int),scope,groups)
        for name,new,old,scope in [
            ("gate_minus_tabstar", (output["gate"]>=.5)==y, tab==y, entire),
            ("adjusted_minus_control", (output["adjusted"]>=.5)==y, (output["control"]>=.5)==y, entire),
            ("density_increment_minus_release_on_common", (baseline^increment)==y, baseline==y, common),
        ]
    }
    candidates = valid & known & (baseline==nine) & (tab==baseline) & (den!=baseline)
    active = fields[:,5]
    buckets = {"no_recorded_spending":active==0,"one_service":active==1,"two_or_more_services":active>=2,
               "solo_order":fields[:,10]==1,"multi_passenger_order":fields[:,10]>1,
               "density_proposes_transported":den,"density_proposes_not_transported":~den}
    result["density_only_disagreement_profile"] = {
        key: dict(passengers=int((candidates & subset).sum()),
                  density_was_right=int((candidates & subset & (den==y)).sum()),
                  old_label_was_right=int((candidates & subset & (den!=y)).sum()))
        for key,subset in buckets.items()
    }
    return result


def report():
    from outside_panel_coverage_n26 import projection
    from arbitration_against_scored_nine_n26 import champions
    raw,y,groups,fold,views,_ = projection(include_full=True)
    nine,known,_,_ = champions()
    with np.load(tabstar.ART/"strict_witnesses.npz") as z:
        strict = np.r_[z["all_wrong"],z["all_correct"]]
    views["strict_consensus"] = np.isin(np.arange(len(y)),strict),views["full"][1]
    fields = case_features(raw,groups)
    predictions = {"matched_six_fold_experts":{}, "refit_seven_fold_experts":{}}
    calibration_counts = {}
    for k in range(8):
        pair = k-k%2
        partner = k^1
        all_query = np.flatnonzero(np.isin(fold,[k,partner]))
        support = np.flatnonzero(~np.isin(fold,[k,partner]))
        a,q = np.flatnonzero(fold==partner),np.flatnonzero(fold==k)
        assert not np.isin(groups[q],groups[a]).any()
        place = OUT/f"pair{pair}_{pair+1}"
        nested = read_experts(raw,y,groups,support,all_query,place/"density.npz",place/f"tuned{pair}.npz")
        cal = nested[fold[all_query]==partner]
        np.testing.assert_array_equal(all_query[fold[all_query]==partner],a)
        np.testing.assert_array_equal(all_query[fold[all_query]==k],q)
        calibration_counts[str(k)] = dict(passengers=len(a),bookings=len(np.unique(groups[a])),
                                         disagreements=int(((cal[:,0]>=.5)!=(cal[:,1]>=.5)).sum()))
        full = read_experts(raw,y,groups,np.flatnonzero(fold!=k),q,density.ART/f"fold{k}.npz",
                            tabstar.ART/f"tuned{k}.npz")
        for tag,expert in [("matched_six_fold_experts",nested[fold[all_query]==k]),
                           ("refit_seven_fold_experts",full)]:
            output = decide(cal,expert,fields[a],fields[q],y[a])
            for n,v in output.items():
                predictions[tag].setdefault(n,np.full(len(y),np.nan))[q] = v
    for tag,cols in predictions.items():
        assert all(np.isfinite(p).all() for p in cols.values())
        np.savez_compressed(OUT/f"{tag}.npz",rows=np.arange(len(y)),fold=fold,**cols)
    metrics = {tag:summarize(y,fold,cols,views,nine,known,fields,groups) for tag,cols in predictions.items()}
    REPORT.write_text(json.dumps(dict(
        protocol="Fold k uses fold k XOR 1 as calibration. Both base experts exclude that entire pair during calibration. "
                 "No release prediction is an input to the learner. Matched uses the same two models on both folds. "
                 "Refit applies the fixed gate to separately trained outer experts using the seven nonquery folds. "
                 "The gate trains only on opposed labels. Two residual log-odds calibrators fit every calibration row, "
                 "with matched raw fields and TabSTAR; the extended calibrator also gets the density odds.",
        calibration=calibration_counts,results=metrics),indent=2)+"\n")
    for tag,m in metrics.items():
        print(tag,m["gate_on_disagreement"],m["shared_release_decisions"]["full"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, choices=[0, 2, 4, 6])
    args = parser.parse_args()
    if args.pair is None:
        report()
    else:
        fit_pair(args.pair)
