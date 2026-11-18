"""Export v115 by combining two independently fitted error correction routes."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent
NAME="proposal_v115_consensus_error_routes"
VERSION="v115_v108_consensus_and_divided_routes"
BASE="proposal_v108_party_bills"
THREE="proposal_v114_error_joint"
MARGIN="full_challenger_margin"
HISTORICAL=["proposal_batch_v44","proposal_nov16_full_episode","proposal_nov16_tabdpt",
            "proposal_nov16_kumo","proposal_online_first_exaone_fixed"]


def official_decisions(name,ids):
    path=ROOT/"artifacts"/name/"submission.csv"
    attempts=list((ROOT/"reports"/name/"online_attempts").glob("*.json"))
    records=[json.loads(f.read_text()) for f in attempts]
    digest=sha256(path)
    assert any(r["file_sha256"]==digest and r["evaluation"]["Status"]=="COMPLETE" for r in records)
    frame=pd.read_csv(path)
    np.testing.assert_array_equal(ids,frame.PassengerId)
    assert frame.Transported.dtype==bool
    return frame.Transported.to_numpy(),digest


def route(base,historical,three,margin):
    assert historical.shape==(len(base),5)
    assert all(a.dtype==bool for a in [base,historical,three,margin])
    agree=np.all(historical==base[:,None],axis=1)
    oppose=np.all(historical!=base[:,None],axis=1)
    divided=~(agree|oppose)
    flip=(agree&three)|(divided&margin)
    return base^flip,agree,divided,oppose


def validate_rule_on_development():
    ids=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv").row_id
    with np.load(ROOT/"artifacts/consensus_error_routes.npz") as z:
        np.testing.assert_array_equal(z["dev"],ids)
        assert int(z["combined_by_historical_agreement"].sum())==87
    row=json.loads((ROOT/"reports/champion_error_portfolio/consensus_routes.json").read_text())["rules"]["combined_by_historical_agreement"]
    assert (row["repairs"],row["breaks"],row["hard546_new"],row["hard_lost"])==(52,35,2,0)


def export():
    validate_rule_on_development()
    ids=pd.read_csv(ROOT/"data/raw/test.csv",usecols=["PassengerId"]).PassengerId
    actual_base,base_sha=official_decisions(BASE,ids)
    bp=pd.read_csv(ROOT/"artifacts"/BASE/"test_probabilities.csv",float_precision="round_trip")
    np.testing.assert_array_equal(ids,bp.PassengerId)
    np.testing.assert_array_equal(bp.probability,bp.reload_probability)
    np.testing.assert_array_equal(bp.probability>=.5,actual_base)
    sources={BASE:base_sha}
    hist=[]
    for name in HISTORICAL:
        decisions,digest=official_decisions(name,ids)
        hist.append(decisions);sources[name]=digest
    h=np.column_stack(hist)
    triple_dir=ROOT/"artifacts"/THREE
    margin_dir=ROOT/"artifacts"/MARGIN
    triple_manifest=json.loads((triple_dir/"manifest.json").read_text())
    margin_manifest=json.loads((margin_dir/"manifest.json").read_text())
    assert sha256(triple_dir/"model.joblib")==triple_manifest["model_sha256"]
    assert sha256(margin_dir/"model.joblib")==margin_manifest["model_sha256"]
    assert sha256(margin_dir/"test_decisions.csv")==margin_manifest["test_decisions_sha256"]
    triples=pd.read_csv(triple_dir/"expert_opinions.csv")
    margins=pd.read_csv(margin_dir/"test_decisions.csv")
    np.testing.assert_array_equal(triples.PassengerId,ids)
    np.testing.assert_array_equal(margins.PassengerId,ids)
    triple=triples.flip.to_numpy()
    triple_vote=(triples[["matched_party","forest_chain","supervised_em"]].to_numpy()>=.5)
    np.testing.assert_array_equal(triple,(triple_vote!=actual_base[:,None]).all(axis=1))
    with np.load(triple_dir/"reload_probabilities.npz") as z:
        triple_reload=z["flip"].astype(bool)
    np.testing.assert_array_equal(triple,triple_reload)
    decision,agree,divided,oppose=route(actual_base,h,triple,margins.margin_flip.to_numpy())
    again,*_=route(actual_base,h,triple_reload,margins.reload_margin_flip.to_numpy())
    np.testing.assert_array_equal(decision,again)
    changed=decision!=actual_base
    # A hard voting rule gives a classification, not a newly calibrated probability.
    scores=np.where(changed,np.where(decision,.75,.25),bp.probability)
    reload_scores=np.where(again!=actual_base,np.where(again,.75,.25),bp.reload_probability)
    np.testing.assert_array_equal(scores,reload_scores)
    np.testing.assert_array_equal(scores>=.5,decision)
    out=ROOT/"artifacts"/NAME;report=ROOT/"reports"/NAME
    out.mkdir(parents=True,exist_ok=False)
    router=dict(version=VERSION,base=BASE,historical=HISTORICAL,
                components={THREE:triple_manifest["model_sha256"],MARGIN:margin_manifest["model_sha256"]},
                input_submissions=sources,three_opinions_sha256=sha256(triple_dir/"expert_opinions.csv"),
                margin_decisions_sha256=sha256(margin_dir/"test_decisions.csv"),
                rule="If all five historical labels equal the base, apply the three-expert veto. If divided, apply margin. Otherwise keep base.")
    write(out/"router.json",router)
    pd.DataFrame(dict(PassengerId=ids,base=actual_base,all_agree=agree,divided=divided,all_oppose=oppose,
        three_experts_flip=triple,margin_flip=margins.margin_flip,Transported=decision)).to_csv(out/"route_trace.csv",index=False)
    pd.DataFrame(dict(PassengerId=ids,probability=scores,reload_probability=reload_scores)).to_csv(out/"test_probabilities.csv",index=False)
    pd.DataFrame(dict(PassengerId=ids,Transported=decision)).to_csv(out/"submission.csv",index=False)
    check=validate_submission(out/"submission.csv",ROOT/"data/raw/test.csv")
    m=dict(**check,version=VERSION,model_sha256=sha256(out/"router.json"),reload_difference=0,
           changed_vs_v108=int(changed.sum()),consensus_cases=int(agree.sum()),divided_cases=int(divided.sum()),
           all_oppose_cases=int(oppose.sum()),changes_consensus=int((agree&changed).sum()),changes_divided=int((divided&changed).sum()),
           route_trace_sha256=sha256(out/"route_trace.csv"),vote_encoding="On switched rows, 0.25/0.75 encodes the output decision; it is not a calibrated probability.",
           dependencies=router["components"],recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    for folder in [out,report]:
        write(folder/"manifest.json",m);write(folder/"selection.json",dict(version=VERSION,threshold=.5))
    print(m,flush=True)


if __name__=="__main__":
    export()
