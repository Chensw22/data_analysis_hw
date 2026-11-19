"""Conditional wrong/right mixtures in independent spending and manifest views."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logsumexp
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import error_case_retrieval as retrieval
import relational_error_corrector as ref
from observation_collision_audit import metrics
from route_consensus_errors import FAMILIES
from stable_consensus_selector import persistent_mask
from teacher_distillation import kumo_inner

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
NAME = "prototype_error_specialist"


def ratios(train, direction, errors, query, qdirection, k, seed):
    """Equal covariance for both error states; mixing weights from their separate fits."""
    result = np.zeros(len(query))
    for vote in [False, True]:
        select = qdirection == vote
        clouds = []
        residual = 0.
        count = 0
        for label in [False, True]:
            x = train[(direction == vote) & (errors == label)]
            if len(x) < k:
                raise ValueError("Insufficient discovery examples for prototype support")
            model = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(x)
            weight = np.bincount(model.labels_, minlength=k) / len(x)
            active = weight > 0
            clouds.append((model.cluster_centers_[active], np.log(weight[active])))
            residual += model.inertia_
            count += x.size
        # The common isotropic variance cancels the normalizing constants in the ratio.
        var = max(residual / count, 1e-4)
        logdens = [logsumexp(w - cdist(query[select], centers, metric="sqeuclidean") / (2*var), axis=1)
                   for centers, w in clouds]
        result[select] = np.clip(logdens[1] - logdens[0], -12., 12.)
    return result


def design(base, witness, signals=()):
    sign = np.where(base >= .5, -1., 1.)
    return np.column_stack([base >= .5, -np.abs(ref.logit(base)), sign * ref.logit(witness), *signals])


def run():
    data, splits, fold, models, evidence = ref.overlap.load_authenticated()
    dev, y = data["dev"], data["y"].astype(bool)
    groups = np.asarray(data["groups"])
    frame = data["features"].iloc[dev].reset_index(drop=True)
    with np.load(ART / "v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"], dev)
        np.testing.assert_array_equal(z["fold_id"], fold)
        base = z["v108"]
    vote = base >= .5
    unanimous = np.all(np.column_stack([models[k] >= .5 for k in FAMILIES]) == vote[:, None], axis=1)
    scores = {k: np.full((len(y), 3), np.nan) for k in
              ["confidence", "one_center", "four_centers", "uncalibrated_four"]}
    pos = pd.Index(dev)
    records = []
    for outer, (a, b) in enumerate(splits):
        kumo = kumo_inner(outer, a, b, dev, data["y"], evidence)
        loc = pd.Index(dev[a])
        for j in range(3):
            with np.load(ART / f"error_evidence_challengers/outer{outer}_packet{j}.npz") as z:
                d, c = pos.get_indexer(z["D_ids"]), pos.get_indexer(z["C_ids"])
                assert min(d.min(), c.min()) >= 0
                np.testing.assert_array_equal(z["outer_query_ids"], dev[b])
                assert not np.intersect1d(z["teacher_support_ids"], np.r_[dev[b], dev[d], dev[c]]).size
                assert not np.intersect1d(groups[d], groups[c]).size
                assert not np.intersect1d(b,np.r_[c,d]).size
                pd0, pc = z["D_base"], z["C_base"]
                kd, kc = kumo[loc.get_indexer(dev[d])], kumo[loc.get_indexer(dev[c])]
            aligned = (pd0 >= .5) == (kd >= .5)
            d, pd0 = d[aligned], pd0[aligned]
            aligned = (pc >= .5) == (kc >= .5)
            c, pc, kc = c[aligned], pc[aligned], kc[aligned]
            db = pd0 >= .5
            err = db != y[d]
            qb = np.r_[pc, base[b]]
            qk = np.r_[kc, models["Kumo"][b]]
            q = np.r_[c,b]
            designs = {}
            seed = 20261119 + 100*outer + j
            for k, name in [(1,"one_center"), (4,"four_centers")]:
                features = []
                for view in ["consumption", "passenger_cabin"]:
                    encoder = retrieval.View(view).fit(frame.iloc[d])
                    t = encoder.transform(frame.iloc[d])
                    query = encoder.transform(frame.iloc[q])
                    features.append(ratios(t, db, err, query, qb >= .5, k, seed))
                designs[name] = design(qb, qk, features)
                if k == 4:
                    priors = [np.mean(err[db == label]) for label in [False, True]]
                    odds = ref.logit(np.asarray(priors)[vote[b].astype(int)])
                    scores["uncalibrated_four"][b,j] = expit(odds + np.mean(features,axis=0)[len(c):])
            designs["confidence"] = design(qb,qk)
            for name, x in designs.items():
                fit = make_pipeline(StandardScaler(), LogisticRegression(C=.15, max_iter=500))
                fit.fit(x[:len(c)], (pc >= .5) != y[c])
                scores[name][b,j] = fit.predict_proba(x[len(c):])[:,1]
            records.append(dict(outer=outer,packet=j,discovery=len(d),errors=int(err.sum()),calibration=len(c)))
    cat = pd.read_csv(ART / "error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cat.row_id,dev)
    np.testing.assert_array_equal(cat.y,y)
    hard = cat.all_recipes_still_wrong.to_numpy(bool)
    persistent = persistent_mask(data, models, hard)
    result = {"protocol": "Within each existing teacher packet select discovery and calibration passengers where rawMitra and Kumo agree. On discovery, separate k-means mixtures by prediction direction and correct/wrong, independently for log-spend and passenger/cabin views. Use independent calibration parties for a logistic error selector. At inference, average three packet probabilities; only flip on a five-family agreement with v108 and risk >=.5. Uncalibrated control uses discovery class prior and equal mean log density ratios. No Name; 546/510 labels only for final scoring; adaptive development set.", "packets": records, "methods": {}}
    arrays = dict(dev=dev, fold=fold, base=vote)
    for name, packets in scores.items():
        assert np.isfinite(packets).all()
        mean = packets.mean(1)
        pred = vote ^ (unanimous & (mean >= .5))
        row = metrics(pred,vote,y,fold,hard)
        row.update(persistent_repaired=int((persistent & (vote!=y) & (pred==y)).sum()),
                   persistent_harmed=int((persistent & (vote==y) & (pred!=y)).sum()))
        result["methods"][name] = row
        arrays[name] = pred
        arrays[name+"_risk"] = mean
    # Match exactly the traffic of the uncalibrated head inside each fold.
    # These ranks only diagnose concentration of errors, not a release threshold.
    result["matched_traffic"] = {}
    for key in ["confidence", "one_center", "four_centers", "uncalibrated_four"]:
        rank = arrays[key + "_risk"]
        selected = np.zeros(len(y), bool)
        for _, b in splits:
            eligible = b[unanimous[b]]
            n = int(np.sum(arrays["uncalibrated_four"][eligible] != vote[eligible]))
            selected[eligible[np.argsort(-rank[eligible], kind="stable")[:n]]] = True
        row = metrics(vote ^ selected, vote, y, fold, hard)
        row["persistent_repaired"] = int((persistent & (vote != y) & selected).sum())
        result["matched_traffic"][key] = row
    ref.write(ROOT / "reports" / NAME / "result.json",result)
    np.savez_compressed(ART / (NAME+".npz"),**arrays)
    print(result["methods"])


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
