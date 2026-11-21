"""Compute an oracle upper bound from eight frozen development decision vectors."""
from math import ceil

import numpy as np
import pandas as pd

from sleeper_consensus_scope import ART
from exact_fold_error_selector import cached
from error_trained_experts import EXPERTS
from relational_error_corrector import overlap, write
from residual_vote_nov21 import decide
from sleeper_manifest_alternatives import decisions, relative_risk
from sleeper_consensus_scope import ROOT


def run():
    inp, _, fold, _, _ = overlap.load_authenticated()
    y, dev = inp["y"].astype(bool), inp["dev"]
    party = pd.read_csv(ART / "nov21_party_negative/oof.csv")
    berth = pd.read_csv(ART / "nov21_case_checks/berth_local_rows.csv")
    bills = pd.read_csv(ART / "nov21_case_checks/likelihood_local_rows.csv")
    raw = pd.read_csv(ROOT / "data/raw/train.csv").iloc[dev].reset_index(drop=True)
    np.testing.assert_array_equal(party.row, dev)
    for frame in [berth, bills]:
        np.testing.assert_array_equal(frame.PassengerId, raw.PassengerId)
    for frame in [party, berth, bills]:
        np.testing.assert_array_equal(frame.fold, fold)
        np.testing.assert_array_equal(frame.target, y)
    def get(path, key):
        return cached(path + ".npz", key, dev, fold)

    base = get("sleeper_consensus_scope", "base")
    relative = relative_risk(inp["features"].iloc[dev], y, np.asarray(inp["groups"]))["party_a"]
    incumbent, sleep = decisions(base, get("sleeper_consensus_scope", "original"),
        get("sleeper_manifest_risk", "manifest_party_a"), get("sleeper_manifest_risk", "manifest_berth_party_a"),
        get("sleeper_consensus_scope", "eligible"), relative)
    for other in [party.v128, berth.incumbent, bills.incumbent]:
        np.testing.assert_array_equal(incumbent, other)
    votes = decide(base, incumbent, get("consensus_error_routes", "margin_selector"),
        (get("error_trained_experts", "experts")[:, EXPERTS.index("matched_party")] >= .5) != base,
        get("consensus_error_routes", "three_challengers"), get("replicate_error_committee","original_committee") != base,
        get("conditional_donor_errors", "matched_donors_risk").mean(1) >= .5)
    combined = np.where(party.family_consensus != incumbent, party.family_consensus,
                        bills.v136_likelihood_bills)
    final = pd.read_csv(ART / "nov21_case_checks/v136_final_oof.csv")
    np.testing.assert_array_equal(final.PassengerId, raw.PassengerId)
    np.testing.assert_array_equal(final.fold, fold)
    np.testing.assert_array_equal(final.true, y)
    np.testing.assert_array_equal(final.v136_final, combined)
    pred = {"v129": sleep["v129"], "v130": sleep["v130"], "v131": party.family_consensus,
            "v132": party.family_contrast, **votes, "v135": berth.v135d, "v136": combined}
    correct = incumbent == y
    changes = np.column_stack([p != incumbent for p in pred.values()])
    fixes = (~correct)[:,None] & changes
    harms = correct[:,None] & changes
    any_fix, any_harm = fixes.any(1), harms.any(1)
    hard, persistent = berth.hard.to_numpy(bool), berth.persistent.to_numpy(bool)
    np.testing.assert_array_equal(hard, party.hard)
    np.testing.assert_array_equal(hard, bills.hard)
    np.testing.assert_array_equal(persistent, bills.persistent)
    target = ceil(.85 * len(y))
    result = dict(versions=list(pred), rows=len(y), incumbent_correct=int(correct.sum()),
                  required_correct=target, required_net=int(target-correct.sum()),
                  oracle_fixed=int(any_fix.sum()), oracle_correct=int(correct.sum()+any_fix.sum()),
                  oracle_accuracy=(correct|any_fix).mean(), different_originally_correct=int(any_harm.sum()),
                  cohort={}, unique_fixes={}, overlap=(fixes.astype(int).T @ fixes.astype(int)).tolist())
    for label, mask in [("historical_common", hard), ("persistent", persistent)]:
        result["cohort"][label] = dict(size=int(mask.sum()), v128_correct=int((mask&correct).sum()),
            v128_wrong=int((mask&~correct).sum()), oracle_additional=int((mask&any_fix).sum()),
            remain_wrong=int((mask&~correct&~any_fix).sum()), exposed_correct=int((mask&any_harm).sum()))
    spending = raw[["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]]
    complete_zero = spending.eq(0).all(axis=1)
    cohort = hard & ~correct & ~any_fix
    result["historical_remaining_profile"] = {"rows": int(cohort.sum()), "slices": {}}
    for name, mask in {
        "Earth": raw.HomePlanet.eq("Earth"), "Europa": raw.HomePlanet.eq("Europa"),
        "Mars": raw.HomePlanet.eq("Mars"), "deck_G": raw.Cabin.str.startswith("G/", na=False),
        "observed_cryo": raw.CryoSleep.eq(True), "observed_awake": raw.CryoSleep.eq(False),
        "complete_zero_bills": complete_zero, "any_positive_bill": spending.gt(0).any(axis=1),
        "Earth_G_cryo_complete_zero": raw.HomePlanet.eq("Earth") & raw.Cabin.str.startswith("G/", na=False)
                                      & raw.CryoSleep.eq(True) & complete_zero,
    }.items():
        result["historical_remaining_profile"]["slices"][name] = {
            "slice_total_dev": int(mask.sum()), "remaining_in_slice": int((cohort & mask).sum())}
    result["persistent_remaining_within_historical_remaining"] = int(
        (persistent & ~correct & ~any_fix & cohort).sum())
    for i, version in enumerate(pred):
        result["unique_fixes"][version] = dict(all=int(fixes[:,i].sum()), hard=int((fixes[:,i]&hard).sum()),
            exclusive=int((fixes[:,i] & (fixes.sum(1)==1)).sum()))
    write(ROOT / "reports/nov21_rescue_coverage/result.json", result)
    print(result)


if __name__ == "__main__":
    run()
