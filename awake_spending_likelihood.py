"""Cross-fit spending-pattern likelihoods for false negative corrections."""
from pathlib import Path
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

import export_consensus_routes as routes
import relational_error_corrector as ref
from confirm_sleeper_scope import NAME as INCUMBENT, confirm
from exact_fold_error_selector import counts
from exaone_validation_data import compact_features
from peer_bill_correction import BILLS
from stable_consensus_selector import persistent_mask
from route_consensus_errors import FAMILIES
from export_nov21_error_batch import test_probabilities

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ART / "nov21_case_checks"


def bins(a, edges):
    return np.searchsorted(edges, a, side="left")


def billing_keys(x):
    # Positive bill or recorded non-sleeper provides direct evidence of wakefulness.
    expenses = x[BILLS].fillna(0)
    safe = expenses.FoodCourt + expenses.ShoppingMall
    services = expenses.RoomService + expenses.Spa + expenses.VRDeck
    total = safe + services
    k = pd.DataFrame(index=x.index)
    k["planet"] = x.HomePlanet
    k["deck"] = x.Deck
    k["benefit"] = bins(safe, [0, 99, 499, 999, 1999])
    k["services"] = bins(services, [0, 99, 499, 999, 1999])
    k["share"] = bins((safe + 10) / (total + 20), [.05,.25,.45,.55,.75,.95])
    k["volume"] = bins(total, [0, 399, 749, 999, 1499, 2999])
    k["age"] = bins(x.Age.fillna(28), [12, 19, 39])
    k["side"] = x.Side
    k["missing"] = x[BILLS].isna().any(axis=1)
    k["source"] = expenses.idxmax(axis=1).where(total.gt(0), "none")
    return k, (x.CryoSleep.ne("True") & total.gt(0)).to_numpy()


def posterior(k, t, q, y):
    train = k.iloc[t].assign(target=y.astype(int))
    query = k.iloc[q]
    def lookup(keys):
        stats = train.groupby(keys).target.agg(["sum","count"])
        indexed = pd.MultiIndex.from_frame(query[keys]) if len(keys)>1 else pd.Index(query[keys[0]])
        hits = stats.reindex(indexed).fillna(0)
        return hits["sum"].to_numpy(), hits["count"].to_numpy()
    def shrink(keys, prior, strength):
        success, n = lookup(keys)
        return (success + strength * prior) / (n + strength), n
    prior,_ = shrink(["planet"], y.mean(), 16)
    a,_ = shrink(["planet","share"], prior, 16)
    a,n1 = shrink(["planet","share","volume"], a, 15)
    b,_ = shrink(["planet","source"], prior, 16)
    b,n2 = shrink(["planet","benefit","services"], b, 15)
    # A matched manifestation of spending conditioned on deck and cabin side.
    c,_ = shrink(["deck","source"], prior, 16)
    c,n3 = shrink(["deck","side","source","age"], c, 20)
    return np.stack([a,b,c,n1,n2,n3],axis=1)


def correct(anchor, base, votes, p, packet, eligible):
    a,b,c,n1,n2,n3 = packet.T
    pooled = (a+b+c)/3
    supported = (n1 >= 10) & (n2 >= 10) & (n3 >= 8)
    neg = (~anchor & ~base & eligible & supported)
    unanimous = (~votes).all(1)
    return {"v136_likelihood": anchor | (neg & unanimous & (pooled >= .58)),
            "v136_likelihood_margin": anchor | (neg & ((a+b)/2 >= .55) & (pooled > .5 + (.5-p))),
            "v136_likelihood_bills": anchor | (neg & (a>=.60) & (b>=.60) & (c>=.50))}


def run():
    inp, _, fold, models, _ = ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"].astype(bool)
    train=pd.read_csv(ROOT/"data/raw/train.csv", usecols=lambda c:c not in ["Name","Transported"])
    test=pd.read_csv(ROOT/"data/raw/test.csv", usecols=lambda c:c!="Name")
    roster=pd.concat([train,test],ignore_index=True)
    groups=roster.PassengerId.str.split("_").str[0].to_numpy()
    k,awake=billing_keys(compact_features(roster))
    evidence=np.zeros((len(dev),6))
    for part in np.unique(fold):
        q=fold==part
        t=~q & awake[dev] & ~np.isin(groups[dev],groups[dev[q]])
        evidence[q]=posterior(k,dev[t],dev[q],y[t])
    with np.load(ART/"sleeper_consensus_scope.npz") as scoped, np.load(ART/"sleeper_manifest_risk.npz") as wit, np.load(ART/"v108_exact_oof/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        np.testing.assert_array_equal(scoped["dev"],dev)
        np.testing.assert_array_equal(wit["dev"],dev)
        anchor=confirm(scoped["base"],scoped["original"],wit["manifest_berth_party_a"])
        p=z["v108"]
        base=p>=.5
    candidates=correct(anchor,base,np.column_stack([models[n]>=.5 for n in FAMILIES]),p,evidence,awake[dev])
    rows=pd.read_csv(ART/"error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(rows.row_id,dev)
    hard=rows.all_recipes_still_wrong.to_numpy(bool)
    persistent=persistent_mask(inp,models,hard)
    result={n:counts(val,anchor,y,fold,hard,persistent) for n,val in candidates.items()}
    party_fold = np.zeros(len(y),int)
    for part, (_,valid) in enumerate(GroupKFold(3).split(dev,y,groups[dev])):
        party_fold[valid] = part
    boarding_fold = np.searchsorted(np.quantile(groups[dev].astype(int), [1/3,2/3]),groups[dev].astype(int))
    repeats = {}
    for partition, alt in [("whole_parties",party_fold),("boarding_blocks",boarding_fold)]:
        packet = np.zeros_like(evidence)
        for part in range(3):
            q = alt==part
            t = ~q & awake[dev] & ~np.isin(groups[dev],groups[dev[q]])
            packet[q]=posterior(k,dev[t],dev[q],y[t])
        preds=correct(anchor,base,np.column_stack([models[n]>=.5 for n in FAMILIES]),p,packet,awake[dev])
        repeats[partition]={n:counts(v,anchor,y,alt,hard,persistent) for n,v in preds.items()}
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(dict(PassengerId=train.PassengerId.iloc[dev].to_numpy(),fold=fold,target=y,incumbent=anchor,
                      hard=hard,persistent=persistent, **candidates)).to_csv(OUT/"likelihood_local_rows.csv",index=False)
    all_y=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["Transported"]).Transported.to_numpy(bool)
    ids=test.PassengerId.to_numpy(str)
    t=np.flatnonzero(awake[:len(train)] & ~np.isin(groups[:len(train)],groups[len(train):]))
    q=np.arange(len(train),len(roster))
    pkt=posterior(k,t,q,all_y[t])
    anchor=routes.official_decisions(INCUMBENT,ids)[0]
    p=test_probabilities(routes.BASE,ids)
    hist=np.column_stack([routes.official_decisions(n,ids)[0] for n in routes.HISTORICAL])
    for n,val in correct(anchor,p>=.5,hist,p,pkt,awake[q]).items():
        result[n]["test_changes_vs_v128"]=int((val!=anchor).sum())
        pd.DataFrame(dict(PassengerId=ids,Transported=val)).to_csv(OUT/(n+".csv"),index=False)
    ref.write(ROOT/"reports/nov21_case_checks/spending_likelihood.json",dict(results=result,alternative_partitions=repeats,description=
        "Positive spending and observed awake or missing cryosleep. Three empirical probability tables per held-out "
        "row, each shrunk toward broad planet or deck feature tables. All parties of query batch excluded. "
        "Flip negative only after minimum counts of 10,10,8 in the three finest cells. "
        "Unanimous 58% pooled, probability margin, and consistent-bill 60/60/50 variants."))
    print(result)


if __name__=="__main__":
    run()
