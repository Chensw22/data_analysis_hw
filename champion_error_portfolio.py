"""Evaluate deployable confirmation rules and counterfactual error coverage on the actual champion."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from error_trained_experts import summarize,EXPERTS
import relational_error_corrector as ref

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"


def compare(decision,base,p,y,hard,fold):
    row=summarize(y,decision,base,hard,fold)
    uncertain=np.zeros(len(y),bool)
    for j in range(3):
        ids=np.flatnonzero(fold==j)
        n=np.sum(decision[ids]!=base[ids])
        order=ids[np.argsort(np.abs(p[ids]-.5),kind="stable")]
        uncertain[order[:n]]=True
    matched=summarize(y,base^uncertain,base,hard,fold)
    assert row["repairs"]+row["breaks"]==int((base!=decision).sum())
    assert row["net"]==int((decision==y).sum()-(base==y).sum())
    row["confidence_only_same_changes"]={k:matched[k] for k in ["repairs","breaks","net","hard546_new"]}
    row["error_direction"]={label:dict(available=int(((base!=y)&mask).sum()),
            repaired=int(((base!=y)&(decision==y)&mask).sum()),
            broken=int(((base==y)&(decision!=y)&mask).sum()))
            for label,mask in [("base_positive",base),("base_negative",~base)]}
    return row


def run():
    exact=ART/"v108_exact_oof/oof.npz"
    assert ref.sha(exact)==json.loads((ROOT/"reports/v108_exact_oof/result.json").read_text())["oof_sha256"]
    with np.load(exact) as z:
        dev,fold,p=z["dev_ids"],z["fold_id"],z["v108"]
    catalog=pd.read_csv(ART/"error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(catalog.row_id,dev);np.testing.assert_array_equal(catalog.fold,fold)
    y=catalog.y.to_numpy(bool);hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    base=p>=.5
    with np.load(ART/"error_case_retrieval/oof.npz") as z:
        np.testing.assert_array_equal(z["dev_ids"],dev)
        compatible=(z["raw"]>=.5)==base
        spend=z["consumption_high_flip"]&compatible
        cabin=z["passenger_cabin_high_flip"]&compatible
        retrieval=z["joint_max_deploy_flip"]&compatible
    with np.load(ART/"error_trained_experts.npz") as z:
        np.testing.assert_array_equal(z["dev"],dev)
        party=(z["experts"][:,EXPERTS.index("matched_party")]>=.5)!=base
    votes={}
    for name,path,key in [("imputation","mixed_type_forest_imputation_oof.npz","forest_chain"),
                          ("latent","latent_class_em_probe/oof.npz","supervised")]:
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z["dev_ids"],dev)
            votes[name]=(z[key]>=.5)!=base
    union=spend|cabin
    split={"spending_only":spend&~cabin,"cabin_only":cabin&~spend,"both":cabin&spend}
    components={"spending_high":spend,"cabin_high":cabin,"either_high":union,"matched_party":party,**votes}
    # These rules consume only a passenger's available predictors at decision time.
    joint={"retrieval_calibrated":retrieval,
           "retrieval_and_imputation":retrieval&votes["imputation"],
           "retrieval_and_latent":retrieval&votes["latent"],
           "matched_party_and_two_challengers":party&votes["imputation"]&votes["latent"],
           "both_retrieval_views_and_two_challengers":spend&cabin&votes["imputation"]&votes["latent"]}
    result={"population":dict(rows=len(y),errors=int((base!=y).sum()),correct=int((base==y).sum()),hard=int(hard.sum()),
                              hard_already_correct=int((hard&(base==y)).sum()),compatible_retrieval_rows=int(compatible.sum())),
            "disjoint_retrieval_contributions":{k:dict(covered_errors=int((m&(base!=y)).sum()),potential_breaks=int((m&(base==y)).sum()),
                          covered_hard_errors=int((m&(base!=y)&hard).sum())) for k,m in split.items()},
            "components":{k:compare(base^m,base,p,y,hard,fold) for k,m in components.items()},
            "joint":{k:compare(base^m,base,p,y,hard,fold) for k,m in joint.items()},
            "notes":"Complete exact champion development predictions. Cases for the retrieval route were learned using inner raw errors. Direction mismatch abstains. All labels are used only after constructing joint query decisions. Rules evaluated on repeatedly used development folds."}
    ref.write(ROOT/"reports/champion_error_portfolio/result.json",result)
    print(result["population"],result["disjoint_retrieval_contributions"])
    print({k:(r["repairs"],r["breaks"],r["net"],r["hard546_new"]) for k,r in result["joint"].items()})


if __name__=="__main__":
    run()
