"""Freeze the observed party-consensus decisions, then apply spending-only additions."""
import numpy as np
import pandas as pd
from awake_spending_likelihood import ART, OUT, ROOT
from confirm_sleeper_scope import NAME as V128
from exact_fold_error_selector import counts
from export_consensus_routes import official_decisions
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

V131 = "proposal_v131_party_consensus"


def combine(v128, v131, bill):
    return np.where(v131 != v128, v131, bill)


def run():
    party_dir = ART / "nov21_party_negative"
    local = pd.read_csv(party_dir / "oof.csv")
    bills = pd.read_csv(OUT / "likelihood_local_rows.csv")
    train_ids = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["PassengerId"]).PassengerId
    np.testing.assert_array_equal(train_ids.iloc[local.row].to_numpy(), bills.PassengerId)
    for left,right in [("v128","incumbent"),("target","target"),("fold","fold"),("hard","hard")]:
        np.testing.assert_array_equal(local[left], bills[right])
    v128 = local.v128.to_numpy()
    v131 = local.family_consensus.to_numpy()
    candidate = bills.v136_likelihood_bills.to_numpy()
    decision = combine(v128, v131, candidate)
    frame = pd.DataFrame(dict(row=local.row, PassengerId=bills.PassengerId, fold=local.fold, true=local.target,
                              v128=v128,v131=v131,v136_final=decision,likelihood_bills=candidate,
                              hard=bills.hard,persistent=bills.persistent))
    frame.to_csv(OUT/"v136_final_oof.csv",index=False)
    evidence = dict(local_overlap=int(((v131!=v128) & (candidate!=v128)).sum()))
    for source,reference in [("v128",v128),("v131",v131)]:
        evidence["vs_"+source]=counts(decision,reference,local.target.to_numpy(),local.fold.to_numpy(),
                                      bills.hard.to_numpy(),bills.persistent.to_numpy())
    test = ROOT / "data/raw/test.csv"
    ids = pd.read_csv(test,usecols=["PassengerId"]).PassengerId.to_numpy()
    v128,hash128 = official_decisions(V128,ids)
    v131,hash131 = official_decisions(V131,ids)
    party = pd.read_csv(party_dir/"family_consensus.csv")
    np.testing.assert_array_equal(ids,party.PassengerId)
    np.testing.assert_array_equal(v131,party.Transported)
    bill_path = OUT/"v136_likelihood_bills.csv"
    bill = pd.read_csv(bill_path)
    np.testing.assert_array_equal(ids,bill.PassengerId)
    candidate = bill.Transported.to_numpy()
    decision = combine(v128,v131,candidate)
    file = OUT/"v136_final.csv"
    pd.DataFrame(dict(PassengerId=ids,Transported=decision)).to_csv(file,index=False)
    manifest = validate_submission(file,test)
    evidence.update(test_overlap=int(((v131!=v128)&(candidate!=v128)).sum()),
                    changed_vs_v131=int((decision!=v131).sum()),changed_vs_v128=int((decision!=v128).sum()),
                    participant_count=len(ids),input_csv_sha256={"v128":hash128,"v131":hash131,"bill":sha256(bill_path)},
                    submission=manifest,local_rows_sha256=sha256(OUT/"v136_final_oof.csv"),
                    reproduce=[".venv/bin/python party_negative_rescue.py",".venv/bin/python awake_spending_likelihood.py",
                               ".venv/bin/python combine_awake_party.py"])
    write(ROOT/"reports/nov21_case_checks/v136_final.json",evidence)
    print(evidence)


if __name__=="__main__":
    run()
