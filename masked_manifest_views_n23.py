"""Learn from entire and partially observed manifests with whole-view masking controls."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


VIEWS = [["HomePlanet","Destination","Deck","Side","CabinNumber"], ["Age","VIP","CryoSleep","PartySize"],
         ["RoomService","Spa","VRDeck"], ["FoodCourt","ShoppingMall"]]


def model(x, y, categorical):
    net = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=17, learning_rate=.065,
            min_samples_leaf=30, l2_regularization=3, categorical_features=categorical,
            early_stopping=False, random_state=271)
    return net.fit(x, y)


def run():
    y, anchor_hits, challengers = read_panel()
    persistent = ~anchor_hits.any(axis=0)
    hard = persistent & ~challengers.any(axis=(0, 1))
    rows = pd.read_csv(ART / "nov23_global_error_inventory.csv", usecols=["row"]).row.to_numpy()
    raw = pd.concat([pd.read_csv(ROOT/f"data/raw/{source}.csv",usecols=lambda c:c not in ["Name","Transported"])
                     for source in ["train","test"]],ignore_index=True)
    group = raw.PassengerId.str.split("_").str[0]
    manifest = compact_features(raw).assign(PartySize=group.map(group.value_counts())).iloc[rows].copy()
    group = group.iloc[rows].to_numpy()
    cat = manifest.columns.isin(CATEGORICAL)
    for col in CATEGORICAL:
        manifest[col] = pd.Categorical(manifest[col]).codes.astype(float)
    x = manifest.to_numpy(float)
    views = [manifest.columns.get_indexer(cols) for cols in VIEWS]
    np.testing.assert_array_equal(np.sort(np.concatenate(views)), np.arange(x.shape[1]))
    out = ART/"nov23_masked_views"
    out.mkdir(exist_ok=True)
    results = {}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as cache:
            fold, base = cache["fold"], cache["base"]
            verify(cache, rows, y, fold)
        variants = {"direct": np.full(len(y), np.nan)}
        for k in np.unique(fold):
            t, q = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            assert not np.intersect1d(group[t],group[q]).size
            variants["direct"][q] = model(x[t], y[t], cat).predict_proba(x[q])[:,1]
            rng = np.random.default_rng(271+int(k))
            assignments = rng.permutation(len(t)) % len(views)
            # Reassign columns to the same sized blocks for the control. Counts,
            # training row weights and number of masked features per row match.
            shuffled = rng.permutation(x.shape[1])
            for name, blocks in [("structured",views),("shuffled",[shuffled[v] for v in views])]:
                augmented = x[t].copy()
                for j, v in enumerate(blocks):
                    augmented[np.ix_(assignments==j,v)] = np.nan
                net = model(np.vstack([x[t],augmented]), np.tile(y[t],2), cat)
                full = net.predict_proba(x[q])[:,1]
                partial = []
                for v in blocks:
                    query = x[q].copy()
                    query[:,v] = np.nan
                    partial.append(net.predict_proba(query)[:,1])
                for mode, score in [("full",full),("marginal",.5*full+.5*np.mean(partial,axis=0))]:
                    key = name+"_"+mode
                    if key not in variants:
                        variants[key] = np.full(len(y), np.nan)
                    variants[key][q] = score
        before = (base>=.5)==y
        metrics = {}
        for name, p in variants.items():
            assert np.isfinite(p).all()
            for key, prob in [(name,p),(name+"_blend",.75*base+.25*p)]:
                ok=(prob>=.5)==y
                metrics[key]=dict(accuracy=float(ok.mean()), fixed=int((ok&~before).sum()),
                                  harmed=int((~ok&before).sum()), persistent1070=int(ok[persistent].sum()),
                                  unresolved466=int(ok[hard].sum()))
        results[seed] = metrics
        np.savez_compressed(out/f"{seed}.npz", row=rows,y=y,fold=fold,base=base,**variants)
        print(seed,[(k,round(v["accuracy"],4),v["unresolved466"]) for k,v in metrics.items()],flush=True)
    write(ROOT/"reports/nov23_masked_views/result.json",dict(views=VIEWS,results=results))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
