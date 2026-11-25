"""Group-held mixed ordinal/nominal outcome densities with Gaussian factor quadrature."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtri
from sklearn.metrics import log_loss

from full_roster_booking_validation_n23 import ROOT, data

ART = ROOT / "artifacts/nov26_ordinal_factor_density"
REPORT = ROOT / "reports/nov23_matched_learning_curve/ordinal_factor_density.json"
BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORIES = ["HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side"]
ORDERED = BILLS + ["Age", "CabinNumber", "BookingSize"]


def features(raw, groups):
    frame = raw.drop(columns=["PassengerId", "Cabin"]).copy()
    frame[["Deck", "CabinNumber", "Side"]] = raw.Cabin.str.split("/", expand=True)
    frame["CabinNumber"] = pd.to_numeric(frame.CabinNumber)
    frame["BookingSize"] = raw.groupby(groups).PassengerId.transform("size")
    return frame


def encode(frame, train):
    """Use common support bins; class-specific empirical masses will set the probit cuts."""
    codes, widths = [], []
    for field in ORDERED:
        values = frame[field].to_numpy(float)
        observed = values[train][np.isfinite(values[train])]
        if field in BILLS:
            edges = np.unique(np.quantile(observed[observed > 0], np.arange(1, 8) / 8))
            # Outcome 0 is missing; outcome 1 is the zero atom; outcomes 2+ are paid bins.
            code = np.where(values == 0, 1, 2 + np.searchsorted(edges, values))
            width = len(edges) + 3
        else:
            edges = np.unique(np.quantile(observed, np.arange(1, 9) / 9))
            code = 1 + np.searchsorted(edges, values)
            width = len(edges) + 2
        codes.append(np.where(np.isfinite(values), code, 0))
        widths.append(width)
    for field in CATEGORIES:
        value = frame[field].astype("string").fillna("__missing__")
        levels = sorted(value.iloc[train].unique())
        codes.append(pd.Categorical(value, categories=levels).codes.astype(int) + 1)
        widths.append(len(levels) + 1)
    x = np.column_stack(codes)
    # Under either class each row contributes exactly one bin in each dimension.
    return np.concatenate([np.eye(w, dtype=np.float32)[v] for v, w in zip(codes, widths)], axis=1), widths


def quadrature(order):
    import torch
    points, masses = np.polynomial.hermite.hermgauss(order)
    grid = np.stack(np.meshgrid(points, points), axis=-1).reshape(-1, 2) * np.sqrt(2.)
    return torch.tensor(grid, dtype=torch.float32), torch.tensor(np.outer(masses, masses).ravel() / np.pi, dtype=torch.float32)


def fit_density(x, widths, train, query, seed):
    import torch
    from torch import nn

    # A tensor-product Gauss-Hermite rule integrates the shared pair of standard normal factors.
    nodes, weight = quadrature(9)
    torch.manual_seed(seed)
    xtr, xq = torch.from_numpy(x[train]), torch.from_numpy(x[query])
    cuts, p_missing, base = [], [], []
    counts = np.split(x[train].sum(axis=0), np.cumsum(widths)[:-1])
    slopes, intercepts = nn.ParameterList(), nn.ParameterList()
    for j, count in enumerate(counts):
        freq = (count + 0.65) / (count.sum() + 0.65 * len(count))
        base.append(np.log(freq))
        if j < len(ORDERED):
            cdf = np.cumsum(freq[1:] / freq[1:].sum())[:-1]
            cuts.append(torch.tensor(ndtri(cdf), dtype=torch.float32))
            p_missing.append(float(freq[0]))
            slopes.append(nn.Parameter(torch.randn(1, 2) * 0.14))
        else:
            slopes.append(nn.Parameter(torch.randn(len(count), 2) * 0.14))
            intercepts.append(nn.Parameter(torch.from_numpy(np.log(freq))))
    optimizer = torch.optim.Adam([*slopes, *intercepts], lr=0.045)
    fixed = np.concatenate(base)
    def probabilities(at_nodes):
        zero = torch.zeros(1, len(at_nodes))
        one = torch.ones(1, len(at_nodes))
        tables = []
        for j, load in enumerate(slopes):
            mean = load @ at_nodes.T
            if j < len(ORDERED):
                # If latent item = (load*z + unit Gaussian noise)/sqrt(1+||load||²),
                # this is its conditional CDF at every class-specific marginal cut.
                cuts_at_nodes = cuts[j][:, None] * torch.sqrt(1 + load.square().sum()) - mean
                cumulative = torch.cat([zero, torch.special.ndtr(cuts_at_nodes), one])
                bins = cumulative.diff(dim=0) * (1 - p_missing[j])
                table = torch.cat([one * p_missing[j], bins]).clamp_min(1e-10)
                tables.append(table.log())
            else:
                tables.append(torch.log_softmax(intercepts[j-len(ORDERED)][:, None] + mean, dim=0))
        return torch.cat(tables)

    for _ in range(85):
        table = probabilities(nodes)
        likelihood = torch.logsumexp(xtr @ table + weight.log(), dim=1)
        penalty = sum(a.square().sum() for a in slopes) * 0.0003
        loss = -likelihood.mean() + penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        has_context = sum(widths[:len(BILLS)])
        result = [x[query] @ fixed]
        for level in (9, 17):
            at_nodes, at_weight = quadrature(level)
            table = probabilities(at_nodes)
            for section in torch.split(table.exp(), widths):
                torch.testing.assert_close(section.sum(dim=0), torch.ones_like(at_weight), atol=2e-6, rtol=0.)
            total = torch.logsumexp(xq @ table + at_weight.log(), dim=1)
            context = torch.logsumexp(xq[:, has_context:] @ table[has_context:] + at_weight.log(), dim=1)
            result.extend([total.numpy(), (total-context).numpy()])
        return np.column_stack(result)


def run():
    import torch
    from catboost import CatBoostClassifier

    torch.set_num_threads(2)
    ART.mkdir(parents=True, exist_ok=True)
    raw, y, groups, folds = data()
    frame = features(raw, groups)
    context = frame.drop(columns=BILLS).copy()
    for field in CATEGORIES:
        context[field] = context[field].fillna("__missing__").astype(str)
    for k in range(8):
        t, q = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert not np.isin(groups[t], groups[q]).any()
        x, widths = encode(frame, t)
        support = [t[y[t] == outcome] for outcome in (False, True)]
        scores = [fit_density(x, widths, a, q, seed=9018+k) for a in support]
        differences = scores[1] - scores[0]
        prior = np.log((len(support[1])+1) / (len(support[0])+1))
        classifier = CatBoostClassifier(iterations=300, depth=5, learning_rate=.055, verbose=False,
                                        random_seed=9018+k, allow_writing_files=False, thread_count=2)
        classifier.fit(context.iloc[t], y[t], cat_features=CATEGORIES)
        context_prior = classifier.predict_proba(context.iloc[q])[:, 1]
        predictions = {
            "independent": expit(prior + differences[:, 0]),
            "joint_nine_nodes": expit(prior + differences[:, 1]),
            "joint": expit(prior + differences[:, 3]),
            "context_only": context_prior,
            "conditional_nine_nodes": expit(logit(context_prior) + differences[:, 2]),
            "conditional_bills": expit(logit(context_prior) + differences[:, 4]),
        }
        np.savez_compressed(ART / f"fold{k}.npz", t=t, q=q, train_y=y[t],
                            passenger=raw.PassengerId.iloc[q].to_numpy(str), **predictions)
        print(k, {n: round(np.mean((p >= .5) == y[q]), 4) for n, p in predictions.items()}, flush=True)


def report():
    from arbitration_against_scored_nine_n26 import champions
    from outside_panel_coverage_n26 import projection

    raw, y, groups, folds, views, _ = projection(include_full=True)
    nine, known, _, _ = champions()
    with np.load(ROOT / "artifacts/nov26_original_tabstar/strict_witnesses.npz") as saved:
        blind_wrong, blind_correct = saved["all_wrong"], saved["all_correct"]
    assert len(blind_wrong) == 178 and len(blind_correct) == 3073
    scores = {}
    tabstar = np.zeros(len(y), dtype=bool)
    for k in range(8):
        with np.load(ART / f"fold{k}.npz") as z:
            t, q = z["t"], z["q"]
            np.testing.assert_array_equal(t, np.flatnonzero(folds != k))
            np.testing.assert_array_equal(q, np.flatnonzero(folds == k))
            np.testing.assert_array_equal(y[t], z["train_y"])
            np.testing.assert_array_equal(raw.PassengerId.iloc[q], z["passenger"])
            assert not np.isin(groups[t], groups[q]).any()
            for name in ("independent", "context_only", "joint", "conditional_bills", "joint_nine_nodes", "conditional_nine_nodes"):
                scores.setdefault(name, np.full(len(y), np.nan))[q] = z[name]
        with np.load(ROOT / f"artifacts/nov26_original_tabstar/tuned{k}.npz") as z:
            np.testing.assert_array_equal(z["q"], q)
            np.testing.assert_array_equal(z["train_y"], y[t])
            tabstar[q] = z["finetuned"] >= .5
    out = {}
    for name, p in scores.items():
        assert np.isfinite(p).all()
        good = (p >= .5) == y
        out[name] = dict(accuracy=float(good.mean()), log_loss=log_loss(y, p),
                         fold_accuracy=[float(good[folds==k].mean()) for k in range(8)], views={},
                         strict=dict(errors=len(blind_wrong), recovered=int(good[blind_wrong].sum()),
                                     correct=len(blind_correct), harmed=int((~good[blind_correct]).sum()),
                                     recovered_rows=blind_wrong[good[blind_wrong]].tolist()))
        for tag, (valid, old) in views.items():
            same = valid & known & (old==nine)
            err = same & (old!=y)
            ok = same & (old==y)
            out[name]["views"][tag] = dict(errors=int(err.sum()), fixed=int((err&good).sum()),
                                           correct=int(ok.sum()), harmed=int((ok&~good).sum()))
            if name in {"joint", "conditional_bills"}:
                confidence = np.maximum(p, 1-p)
                action = old != (p>=.5)
                for label, mask in (("both_density_and_tabstar_dissent", tabstar!=old),
                                    ("density_dissents_tabstar_agrees", tabstar==old)):
                    chosen = same & mask & action
                    out[name]["views"][tag][label] = dict(fixed=int((err & chosen).sum()),
                                                          harmed=int((ok & chosen).sum()))
                out[name]["views"][tag]["confident_disagreements"] = {
                    str(level): dict(fixed=int((err & good & (confidence>=level)).sum()),
                                     harmed=int((ok & ~good & (confidence>=level)).sum()))
                    for level in (.7, .85, .95)
                }
    REPORT.write_text(json.dumps(dict(
        protocol="Eight outer partitions hold out whole booking groups, audited against train_y and PassengerId. "
                 "Two class-specific ordinal probit factors, 9x9 Gauss-Hermite nodes for fitting, 85 Adam steps. "
                 "Joint and conditional predictions use 17x17 nodes; nine-node versions check integration accuracy. "
                 "Positive bill cutpoints are pooled development quantiles. Zero and missing have explicit mass. "
                 "Nominal probabilities use factor-dependent logits. Conditional bill posterior replaces the "
                 "joint model's own nonbill posterior with CatBoost fitted on nonbill covariates.",
        results=out), indent=2) + "\n")
    for n, rec in out.items():
        print(n, rec["accuracy"], rec["views"]["full"], rec["strict"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    if parser.parse_args().report:
        report()
    else:
        run()
