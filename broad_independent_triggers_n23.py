"""Fit independent latent triggers on the same leaf features as an additive-logit control."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, CATEGORICAL, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT / "reports/nov23_independent_triggers"


def basis(frame, y, train):
    panels = [["Deck", "CabinNumber", "Side", "HomePlanet", "Destination", "BookingSize"],
              [*BILLS, "CryoSleep"], ["CryoSleep", "VIP", "Age", "HomePlanet", "BookingSize"],
              list(frame.columns)]
    blocks = []
    for fields in panels:
        model = CatBoostClassifier(iterations=48, depth=3, learning_rate=.12, random_seed=84,
                                   thread_count=2, verbose=False, allow_writing_files=False)
        model.fit(frame.iloc[train][fields], y[train], cat_features=[f for f in fields if f in CATEGORICAL])
        leaves = model.calc_leaf_indexes(frame[fields], thread_count=2)
        columns = (leaves + 8*np.arange(leaves.shape[1])).ravel()
        rows = np.repeat(np.arange(len(frame)), leaves.shape[1])
        blocks.append(csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(frame), 8*leaves.shape[1])))
    return blocks


def linked(z, link):
    """Return response probability and derivatives of the log odds by view."""
    if link == "logit":
        return expit(z.sum(1)), np.ones_like(z)
    # Each view can independently trigger one event with probability sigmoid(z + shift).
    # Set the initial total trigger probability to 1/2 for any number of views.
    shifted = z + np.log(np.expm1(np.log(2) / z.shape[1]))
    hazard = np.logaddexp(0, shifted).sum(1)
    event = -np.expm1(-hazard)
    odds_derivative = expit(shifted) / event[:, None]
    if link == "transport_trigger":
        return event, odds_derivative
    return np.exp(-hazard), -odds_derivative


def fit(basis_train, basis_query, y, link):
    widths = [x.shape[1] for x in basis_train]
    ends = np.cumsum(widths)[:-1]
    def score(blocks, w):
        return np.column_stack([x @ p for x, p in zip(blocks, np.split(w, ends))])
    def objective(w):
        p, derivative = linked(score(basis_train, w), link)
        loss = -np.where(y, np.log(p.clip(1e-14, 1)), np.log1p(-p.clip(0, 1-1e-14))).sum()
        residual = (p-y)[:, None] * derivative
        gradient = np.concatenate([x.T @ r for x, r in zip(basis_train, residual.T)])
        return loss + 5*w@w, gradient + 10*w
    opt = minimize(objective, np.zeros(sum(widths)), method="L-BFGS-B", jac=True,
                   options=dict(maxiter=240, ftol=1e-8))
    if not opt.success:
        raise RuntimeError(opt.message)
    return linked(score(basis_query, opt.x), link)[0]


def main():
    row, y, frame, groups, _ = load()
    with np.load(ART / "nov23_release_vote_information/panel.npz") as z:
        np.testing.assert_array_equal(row, z["row"])
        np.testing.assert_array_equal(y, z["y"])
        wrong = (z["votes"] != y[None, :, None]).all((0, 2))
        right = (z["votes"] == y[None, :, None]).all((0, 2))
    links = ["logit", "transport_trigger", "remain_trigger"]
    rng = np.random.default_rng(144)
    z = rng.normal(size=(13, 4))
    for link in links:
        p, derivative = linked(z, link)
        for j in range(4):
            step = np.zeros_like(z)
            step[:, j] = 1e-5
            fd = (logit(linked(z+step, link)[0]) - logit(linked(z-step, link)[0]))/2e-5
            np.testing.assert_allclose(fd, derivative[:, j], atol=1e-7)
    out = ART / "nov23_independent_triggers"
    out.mkdir(exist_ok=True)
    bank, plans = {}, {}
    for seed in [*map(str, SEEDS), "v131"]:
        fold, ref, opinions, _ = read_opinions(seed, row, y)
        preds = {k: np.full(len(y), np.nan) for k in links}
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            t = np.flatnonzero((fold != f) & ~np.isin(groups, groups[q]))
            assert not np.intersect1d(groups[t], groups[q]).size
            blocks = basis(frame, y, t)
            fit_blocks, query_blocks = [x[t] for x in blocks], [x[q] for x in blocks]
            for key in links:
                preds[key][q] = fit(fit_blocks, query_blocks, y[t], key)
        preds["dual_mean"] = (preds["transport_trigger"] + preds["remain_trigger"])/2
        all_votes = {"reference": ref}
        locked = ref != (opinions[:, 0] >= .5)
        for key, p in preds.items():
            assert np.isfinite(p).all()
            all_votes[key] = p >= .5
            all_votes["quarter_"+key] = np.where(locked, ref, .75*opinions[:, 0]+.25*p >= .5)
        plans[seed] = {"vs_reference": {k: paired(v == y, ref == y) for k, v in all_votes.items()},
                       "vs_logit": {k: paired((p >= .5) == y, all_votes["logit"] == y) for k,p in preds.items()},
                       "log_loss": {k: float(log_loss(y, p)) for k, p in preds.items()}}
        if seed != "v131":
            for k,v in all_votes.items():
                bank.setdefault(k, []).append(v == y)
        np.savez_compressed(out/f"{seed}.npz", row=row,y=y,fold=fold,reference=ref,**preds)
        print(seed, plans[seed]["vs_reference"], flush=True)
    ref_hit = np.stack(bank["reference"])
    stable = {}
    for key, v in bank.items():
        hit = np.stack(v)
        stable[key] = dict(common_wrong_stable_correct=int((hit.all(0) & wrong).sum()),
                           common_right_stable_wrong=int(((~hit).all(0) & right).sum()),
                           mean_gain95_pp=interval((hit.astype(float)-ref_hit).mean(0), groups))
    write(REPORT/"results.json", dict(plans=plans, stable=stable, common_wrong=int(wrong.sum()),
                                      common_right=int(right.sum())))
    print(stable, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
