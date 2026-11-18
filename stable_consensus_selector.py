"""Pool within-outerfold calibration packets to arbitrate dissent against strong consensus."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import second_head_arbitration as arbitration
import sequential_rescue_local as seq
from observation_collision_audit import BILLS, metrics
from route_consensus_errors import FAMILIES

ART = seq.ART
ROOT = seq.ROOT
MODES = ["pooled_margins", "plus_challengers", "plus_context"]


def make_context(inputs):
    x = inputs["features"].iloc[inputs["dev"]].reset_index(drop=True)
    money = x[BILLS]
    block = pd.DataFrame(dict(age=x.Age.fillna(-1), age_missing=x.Age.isna(),
                              npositive=money.gt(0).sum(axis=1), nmissing=money.isna().sum(axis=1)))
    return pd.concat([block, np.log1p(money).fillna(-1),
                      pd.get_dummies(x[["CryoSleep","HomePlanet","Deck"]],dtype=float)],
                     axis=1).to_numpy(float)


def persistent_mask(inputs, models, hard):
    dev, y = inputs["dev"], inputs["y"]
    result = hard.copy()
    for family, key in [("mitra","rawMitra"),("v44","v44")]:
        result &= (models[key] >= .5) != y
        for seed in [20261113,20261114,20261112,20261115,20261116]:
            if seed in (20261113,20261114):
                path = ART / "top5_reference_audit" / family / f"random_{seed}.npz"
            else:
                summary = json.loads((ROOT/"reports/top5_group_phase/refs"/family/"summary.json").read_text())
                row = next(r for r in summary["results"] if r["seed"] == seed)
                path = ROOT / row["oof_path"]
                assert seq.ref.sha(path) == row["oof_sha256"]
            with np.load(path) as z:
                np.testing.assert_array_equal(z["dev_ids"],dev)
                result &= (z["prob"] >= .5) != y
    assert result.sum() == 510
    return result


def design(base, heads, kumo, challengers, ctx):
    basic,_ = seq.gate_features(base,np.column_stack([heads,kumo]))
    enriched,_ = seq.gate_features(base,np.column_stack([heads,kumo,challengers]))
    return {"pooled_margins":basic, "plus_challengers":enriched,
            "plus_context":np.column_stack([enriched,ctx])}


def eligible(proxy, kumo, heads):
    return ((proxy >= .5) == (kumo >= .5)) & ((heads[:,1] >= .5) != (proxy >= .5))


def run():
    inp,splits,fold,models,evidence = seq.ref.overlap.load_authenticated()
    dev,y,groups = inp["dev"],inp["y"].astype(bool),np.asarray(inp["groups"])
    loc = pd.Series(np.arange(len(dev)),index=dev)
    with np.load(ART/"v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        raw = z["v108"]
    with np.load(ART/"confirm_minority_exceptions.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        base,anchor = z["base"],z["second_confirms"]
    with np.load(ART/"sequential_rescue_local.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        heads_saved = z["member"][:,:,[0,2]]
    with np.load(ART/"error_evidence_challengers/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        challengers = z["members"]
    with np.load(ART/"consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        prior_changes = z["combined_by_historical_agreement"]
        np.testing.assert_array_equal(base,(raw>=.5)^prior_changes)
    on = np.column_stack([(models[k] >= .5) == (raw >= .5) for k in FAMILIES]).all(1)
    on &= ~prior_changes & (anchor == base) & ((heads_saved[:,:,1].mean(1)>=.5) != (raw>=.5))
    ctx = make_context(inp)
    frames,cats = seq.experts.views(inp)
    decisions = {k:anchor.copy() for k in MODES}
    stats = []
    for i,(a,b) in enumerate(splits):
        ktrain = arbitration.teachers.kumo_inner(i,a,b,dev,inp["y"],evidence)
        pos = pd.Series(np.arange(len(a)),index=dev[a])
        train_data,query_data,truth,active,training_ids = [],[],[],[],[]
        for j in range(3):
            with np.load(ART/f"error_evidence_challengers/outer{i}_packet{j}.npz") as z:
                d,c = [loc.loc[z[k]].to_numpy() for k in ("D_ids","C_ids")]
                np.testing.assert_array_equal(z["outer_query_ids"],dev[b])
                assert not np.intersect1d(groups[d],groups[c]).size
                assert not np.intersect1d(dev[b],np.r_[dev[d],dev[c],z["teacher_support_ids"]]).size
                assert not np.intersect1d(np.r_[dev[d],dev[c]],z["teacher_support_ids"]).size
                h = arbitration.reproduce_heads(frames,cats,y,groups,d,np.r_[c,b],z["D_base"],20261118+100*i+j)
                np.testing.assert_array_equal(h[len(c):],heads_saved[b,j])
                k = ktrain[pos.loc[z["C_ids"]]]
                cbase = z["C_base"]
                take = eligible(cbase,k,h[:len(c)])
                block = design(cbase,h[:len(c)],k,z["C_challengers"],ctx[c])
                train_data.append({m:v[take] for m,v in block.items()})
                truth.extend(((cbase>=.5)!=y[c])[take])
                training_ids.extend(c[take])
            query_data.append(design(raw[b],h[len(c):],models["Kumo"][b],challengers[b,j],ctx[b]))
            active.append(eligible(raw[b],models["Kumo"][b],h[len(c):]))
        assert len(set(training_ids)) == len(training_ids)
        has = np.stack(active)
        allowed = on[b] & has.any(0)
        stats.append(dict(fold=i, fit=len(truth), wrong=int(sum(truth)), eligible=int(allowed.sum())))
        for m in MODES:
            lr = make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=500))
            lr.fit(np.vstack([d[m] for d in train_data]),truth)
            risk = np.stack([lr.predict_proba(d[m])[:,1] for d in query_data])
            decisions[m][b] ^= allowed & (((risk*has).sum(0)/np.maximum(has.sum(0),1)) >= .5)
    cat = pd.read_csv(ART/"error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,cat.row_id)
    hard = cat.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(inp,models,hard)
    def score(answer,reference):
        detail = metrics(answer,reference,y,fold,hard)
        detail.update(persistent_repaired=int((persistent & (answer==y) & (reference!=y)).sum()),
                      persistent_harmed=int((persistent & (answer!=y) & (reference==y)).sum()),
                      persistent_correct=int((persistent & (answer==y)).sum()))
        return detail
    result = dict(folds=stats, anchor=score(anchor,base),
                  over_second_confirms={m:score(p,anchor) for m,p in decisions.items()},
                  over_v115={m:score(p,base) for m,p in decisions.items()})
    seq.ref.write(ROOT/"reports/stable_consensus_selector/result.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    run()
