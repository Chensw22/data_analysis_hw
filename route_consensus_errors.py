"""Use different evidence for old-model consensus failures and ordinary model disagreements."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import relational_error_corrector as ref
from champion_error_portfolio import compare
from error_trained_experts import EXPERTS

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
FAMILIES=["v44","v81","TabDPT","Kumo","EXAONE"]


def load_packet(name,dev,key="dev_ids"):
    data=np.load(ART/name,allow_pickle=False)
    np.testing.assert_array_equal(data[key],dev)
    return data


def cluster_interval(changed,base,y,groups):
    _,inv=np.unique(groups,return_inverse=True)
    gain=((base^changed)==y).astype(int)-(base==y).astype(int)
    net=np.bincount(inv,weights=gain)
    sizes=np.bincount(inv);rng=np.random.default_rng(20261118)
    values=[]
    for _ in range(20):
        ids=rng.integers(len(net),size=(100,len(net)))
        values.extend((100*net[ids].sum(1)/sizes[ids].sum(1)).tolist())
    return np.quantile(values,[.025,.975]).tolist()


def run():
    inp,splits,fold,bases,_=ref.overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"].astype(bool)
    with load_packet("v108_exact_oof/oof.npz",dev) as z:
        p=z["v108"];np.testing.assert_array_equal(z["fold_id"],fold)
    base=p>=.5
    historical=np.column_stack([bases[n]>=.5 for n in FAMILIES])
    all_same=(historical==base[:,None]).all(axis=1)
    all_opposed=(historical!=base[:,None]).all(axis=1)
    divided=~(all_same|all_opposed)
    with load_packet("error_trained_experts.npz",dev,"dev") as z:
        matched=(z["experts"][:,EXPERTS.index("matched_party")]>=.5)!=base
        pure_gate=(z["v108_risk"]>=.5)&np.any((z["experts"]>=.5)!=base[:,None],axis=1)
    with load_packet("mixed_type_forest_imputation_oof.npz",dev) as z: first=(z["forest_chain"]>=.5)!=base
    with load_packet("latent_class_em_probe/oof.npz",dev) as z: second=(z["supervised"]>=.5)!=base
    with load_packet("error_evidence_challengers/v108_replay.npz",dev) as z:
        np.testing.assert_array_equal(z["v108"],p)
        margin=z["margin_flip"].astype(bool);case=z["case_flip"].astype(bool)
    three=matched&first&second
    votes=np.column_stack([case,pure_gate,three]).sum(axis=1)
    masks={"three_challengers":three,"margin_selector":margin,"any_exception_route":votes>=1,
           "two_exception_routes":votes>=2,"exception_in_historical_consensus":all_same&three,
           "margin_in_divided_histories":divided&margin,
           "combined_by_historical_agreement":(all_same&three)|(divided&margin)}
    catalog=pd.read_csv(ART/"error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(dev,catalog.row_id)
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    # In this diagnostic population all of the historical models were wrong by construction.
    assert np.all(historical[hard]!=y[hard,None])
    result={"population":{k:dict(rows=int(m.sum()),base_errors=int((m&(base!=y)).sum()),
                                      hard=int((m&hard).sum())) for k,m in
            [("historical_all_agree_with_v108",all_same),("historical_divided",divided),("historical_all_oppose_v108",all_opposed)]},
            "components":FAMILIES,"rules":{}}
    for name,change in masks.items():
        d=compare(base^change,base,p,y,hard,fold)
        d["hard_lost"]=int((hard&change&(base==y)).sum())
        d["paired_party_bootstrap_pp"]=cluster_interval(change,base,y,inp["groups"])
        result["rules"][name]=d
        print(name, d["repairs"],d["breaks"],d["net"],d["fold_net"],d["hard546_new"],d["hard_lost"],flush=True)
    np.savez_compressed(ART/"consensus_error_routes.npz",dev=dev,fold=fold,**masks)
    ref.write(ROOT/"reports/champion_error_portfolio/consensus_routes.json",result)


if __name__=="__main__":
    run()
