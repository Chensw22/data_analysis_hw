"""Calibrate exclusively second-head challenges, testing independent challenger margins."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sequential_rescue_local as seq
import teacher_distillation as teachers

ART = seq.ART
OUT = seq.ROOT / "reports/second_head_arbitration"


def reproduce_heads(frames, cats, y, groups, d, query, teacher, seed):
    wrong = (teacher >= .5) != y[d]
    weight = 1. + 2 * wrong
    oof = np.full(len(d), np.nan)
    for t, v in GroupKFold(3).split(d, groups=groups[d]):
        assert not np.intersect1d(groups[d[t]], groups[d[v]]).size
        oof[v] = seq.predict(frames["bills"], cats, y, d[t], d[v], weight[t], seed)
    assert np.isfinite(oof).all()
    misses = wrong & ((oof >= .5) != y[d])
    return np.column_stack([seq.predict(frames["bills"], cats, y, d, query, weight, seed),
                            seq.predict(frames["matched_party"], cats, y, d, query, weight + 4 * misses, seed)])


def exclusive(base, heads):
    contrary = (heads >= .5) != (base[:, None] >= .5)
    return ~contrary[:, 0] & contrary[:, 1]


def run():
    inputs, splits, folds, bases, evidence = seq.ref.overlap.load_authenticated()
    dev, y, groups = inputs["dev"], inputs["y"].astype(bool), np.asarray(inputs["groups"])
    seeds = ART / "sequential_rescue_local.npz"
    expected = json.loads((seq.OUT / "result.json").read_text())["oof_sha256"]
    assert seq.ref.sha(seeds) == expected
    with np.load(seeds) as z:
        np.testing.assert_array_equal(z["dev"], dev)
        np.testing.assert_array_equal(z["folds"], folds)
        first, previous = z["first_only"], z["member"][:, :, [0, 2]]
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        raw = z["v108"]
    with np.load(ART / "consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"], dev)
        protected = z["combined_by_historical_agreement"]
    base = (raw >= .5) ^ protected
    with np.load(ART / "error_evidence_challengers/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        challengers = z["members"]
    protocol = {
        "eligible": "Original v115 and first-only changes protected. Aggregate first expert supports raw direction, second opposes; use calibrated votes only from packets with same exclusive challenge.",
        "regression": "Each packet: restrict disjoint C to first head supports raw teacher and sequential second opposes. Predict teacher error via standardized logistic C=.15 on original margins. Enriched model adds forest-chain CatBoost, latent EM and Kumo signed logits. Match all else.",
        "decision": "Take mean over active packets only. Require >=.5 and at least one active packet. Compare both fixed recipes on all original development rows; no outer-label fitting or thresholds.",
        "limitations": "Kumo uses original inner teacher support in calibration but full outertrain when scoring an outer query; v108 is also a stronger baseline at inference. Reused adaptive 6954 rows.",
        "seed_run": expected,
    }
    seq.ref.write(OUT / "protocol.json", protocol)
    frames, cats = seq.experts.views(inputs)
    loc = pd.Series(np.arange(len(dev)), index=dev)
    risks = {v: np.zeros((len(dev), 3)) for v in ("margins", "with_challengers")}
    active = np.zeros((len(dev), 3), dtype=bool)
    logs = []
    for i, (a, b) in enumerate(splits):
        kumo = teachers.kumo_inner(i, a, b, dev, inputs["y"], evidence)
        cal_pos = pd.Series(np.arange(len(a)), index=dev[a])
        for j in range(3):
            with np.load(ART / f"error_evidence_challengers/outer{i}_packet{j}.npz") as z:
                d, c = [loc.loc[z[k]].to_numpy() for k in ("D_ids", "C_ids")]
                np.testing.assert_array_equal(z["outer_query_ids"], dev[b])
                assert not np.intersect1d(np.r_[dev[d],dev[c],dev[b]], z["teacher_support_ids"]).size
                assert not np.intersect1d(groups[d], groups[c]).size
                heads = reproduce_heads(frames, cats, y, groups, d, np.r_[c, b], z["D_base"], 20261118 + 100*i + j)
                np.testing.assert_array_equal(heads[len(c):], previous[b, j])
                cc = np.column_stack([z["C_challengers"], kumo[cal_pos.loc[z["C_ids"]]]])
                qc = np.column_stack([challengers[b, j], bases["Kumo"][b]])
                cbase = z["C_base"]
            keep = exclusive(cbase, heads[:len(c)])
            active[b, j] = exclusive(raw[b], heads[len(c):])
            wrong = (cbase >= .5) != y[c]
            logs.append(dict(fold=i, packet=j, eligible_C=int(keep.sum()), errors_C=int(wrong[keep].sum())))
            for key in risks:
                pc, pq = heads[:len(c)], heads[len(c):]
                if key == "with_challengers":
                    pc, pq = np.column_stack([pc, cc]), np.column_stack([pq, qc])
                xc, _ = seq.gate_features(cbase, pc)
                xq, _ = seq.gate_features(raw[b], pq)
                model = make_pipeline(StandardScaler(), LogisticRegression(C=.15))
                model.fit(xc[keep], wrong[keep])
                risks[key][b, j] = np.where(active[b, j], model.predict_proba(xq)[:, 1], 0.)
    eligible = exclusive(raw, previous.mean(1)) & ~protected & (first == base) & active.any(1)
    decisions = {k: first ^ (eligible & (v.sum(1) / np.maximum(1, active.sum(1)) >= .5)) for k,v in risks.items()}
    catalog = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id, dev)
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    assert hard.sum() == 546
    def score(p, anchor):
        s = seq.experts.summarize(y,p,anchor,hard,folds)
        s["hard_lost"] = int((hard & (anchor == y) & (p != y)).sum())
        return s
    result = dict(eligible=int(eligible.sum()), eligible_wrong=int((eligible & (first != y)).sum()),
                  eligible_hard_wrong=int((eligible & hard & (first != y)).sum()), training=logs,
                  over_first={k:score(p,first) for k,p in decisions.items()},
                  over_v115={k:score(p,base) for k,p in decisions.items()})
    cache = ART / "second_head_arbitration.npz"
    np.savez_compressed(cache,dev=dev,folds=folds,eligible=eligible,**decisions)
    result["oof_sha256"] = seq.ref.sha(cache)
    seq.ref.write(OUT / "result.json", result)
    print(result["over_first"],flush=True)


if __name__ == "__main__":
    run()
