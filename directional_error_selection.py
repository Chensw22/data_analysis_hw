"""Fixed low-complexity correction rules for false-negative and false-positive consensus."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import prototype_error_specialist as proto

ART, ROOT, ref = proto.ART, proto.ROOT, proto.ref


def gate(x, y, q):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=500))
    return model.fit(x,y).predict_proba(q)[:,1]


def run():
    inputs, splits, folds, models, evidence = ref.overlap.load_authenticated()
    dev, y = inputs["dev"], inputs["y"].astype(bool)
    groups = np.asarray(inputs["groups"])
    frame = inputs["features"].iloc[dev].reset_index(drop=True)
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(dev,z["dev_ids"])
        np.testing.assert_array_equal(folds,z["fold_id"])
        prob = z["v108"]
    vote = prob >= .5
    eligible = (np.column_stack([models[k] >= .5 for k in proto.FAMILIES]) == vote[:,None]).all(1)
    risk = {k:np.full((len(y),3),np.nan) for k in ["shared","separate"]}
    pos = pd.Index(dev)
    counts = []
    for outer,(a,b) in enumerate(splits):
        kumo = proto.kumo_inner(outer,a,b,dev,inputs["y"],evidence)
        aloc = pd.Index(dev[a])
        for j in range(3):
            with np.load(ART / f"error_evidence_challengers/outer{outer}_packet{j}.npz") as z:
                d,c = pos.get_indexer(z["D_ids"]),pos.get_indexer(z["C_ids"])
                np.testing.assert_array_equal(z["outer_query_ids"],dev[b])
                assert not np.intersect1d(z["teacher_support_ids"],np.r_[dev[b],dev[d],dev[c]]).size
                assert (aloc.get_indexer(np.r_[dev[d],dev[c]])>=0).all()
                dp,cp = z["D_base"],z["C_base"]
                dk,ck = kumo[aloc.get_indexer(dev[d])],kumo[aloc.get_indexer(dev[c])]
            active = ((dk>=.5)==(dp>=.5)) & ~np.isin(groups[d],groups[b])
            d,dp = d[active],dp[active]
            active = ((ck>=.5)==(cp>=.5)) & ~np.isin(groups[c],groups[b])
            c,cp,ck = c[active],cp[active],ck[active]
            assert not np.intersect1d(groups[d],groups[c]).size
            assert not np.intersect1d(groups[np.r_[c,d]],groups[b]).size
            q = np.r_[c,b]
            qb = np.r_[cp,prob[b]]
            features = []
            for view in ["consumption","passenger_cabin"]:
                encoder = proto.retrieval.View(view).fit(frame.iloc[d])
                features.append(proto.ratios(encoder.transform(frame.iloc[d]),dp>=.5,(dp>=.5)!=y[d],
                    encoder.transform(frame.iloc[q]),qb>=.5,4,20261119+100*outer+j))
            x = proto.design(qb,np.r_[ck,models["Kumo"][b]],features)
            truth = (cp>=.5)!=y[c]
            cut = len(c)
            risk["shared"][b,j] = gate(x[:cut],truth,x[cut:])
            for direction in [False,True]:
                train = (cp>=.5)==direction
                query = vote[b]==direction
                # First column encodes direction, constant within each separate selector.
                risk["separate"][b[query],j] = gate(x[:cut][train,1:],truth[train],x[cut:][query,1:])
                counts.append(dict(outer=outer,packet=j,positive=direction,cal_rows=int(train.sum()),
                                   cal_errors=int(truth[train].sum())))
    cat = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,cat.row_id)
    np.testing.assert_array_equal(y,cat.y)
    hard = cat.all_recipes_still_wrong.to_numpy(bool)
    persistent = proto.persistent_mask(inputs,models,hard)
    results = dict(calibration=counts,cohorts={},methods={},notes="Original 6954 repeatedly used dev rows. Each discovery and calibration party is excluded from the outer evaluation parties, and both sets are mutually party-disjoint. Frozen rawMitra/Kumo themselves use the historic passenger splits. Calibration uses raw/Kumo agreement as a proxy for five-family agreement with the deployed v108. Both direction-specific .5 rules fixed before scoring. Ranking statistics compare retrospective hard errors with genuinely correct consensus passengers; average precision is prevalence-sensitive.")
    cache = dict(dev=dev,fold=folds,base=vote,eligible=eligible)
    for lab,dr in [("false_negative",False),("false_positive",True)]:
        m = eligible & (vote == dr)
        wrong = m & (vote!=y)
        right = m & (vote==y)
        results["cohorts"][lab] = dict(rows=int(m.sum()),wrong=int(wrong.sum()),correct=int(right.sum()),
            historic_errors=int((hard & wrong).sum()),persistent_errors=int((persistent & wrong).sum()))
        for k,v in risk.items():
            assert np.isfinite(v).all()
            score = v.mean(1)
            change = m & (score>=.5)
            pred = vote ^ change
            row = proto.metrics(pred,vote,y,folds,hard)
            row["persistent_repairs"] = int((persistent & wrong & change).sum())
            row["persistent_harms"] = int((persistent & right & change).sum())
            row["max_risk"] = float(score[m].max())
            for name,t in [("all",wrong),("hard",wrong & hard),("persistent",wrong & persistent)]:
                use = right | t
                row[name+"_AUC"] = float(roc_auc_score(t[use],score[use]))
                row[name+"_AP"] = float(average_precision_score(t[use],score[use]))
            results["methods"][k+"_"+lab] = row
            cache[k+"_"+lab] = pred
            cache[k+"_risk"] = score
    ref.write(ROOT/"reports/directional_error_selection/result.json",results)
    np.savez_compressed(ART/"directional_error_selection.npz",**cache)
    print(results["cohorts"],results["methods"])


if __name__=="__main__":
    with threadpool_limits(limits=2):
        run()
