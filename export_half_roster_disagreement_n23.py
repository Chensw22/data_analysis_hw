"""Export the pooled linear disagreement rule over unmodified v96 decisions in v131."""
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from export_budget_quarter_n23 import source
from full_roster_booking_validation_n23 import ROOT, data, predictor
from full_roster_metric_neighbors_n23 import predict_neighbors, roster_tensor
from half_roster_disagreement_eval_n23 import DEST, NEIGHBOR, inputs
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

VERSION = 'n23_half_roster_disagreement'
NAME = 'proposal_'+VERSION


def calibrator(raw, y, g, fold):
    panel = fold % 2
    with np.load(DEST/'others.npz') as z:
        for k, v in [('passenger', raw.PassengerId), ('panel', panel), ('y', y)]:
            np.testing.assert_array_equal(z[k], v)
        p = np.column_stack([np.zeros(len(y)), *(z[k] for k in ['cat', 'budget', *NEIGHBOR])])
    for h in (0, 1):
        t, q = np.flatnonzero(panel != h), np.flatnonzero(panel == h)
        with np.load(DEST/f'mitra_{h}.npz') as z:
            for k, v in [('train', t), ('query', q), ('labels', y[t]), ('steps', 15)]:
                np.testing.assert_array_equal(z[k], v)
            p[q, 0] = z['p']
    x, ref, choice = inputs(p, raw, g)
    use = ref != choice
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.10, max_iter=500))
    model.fit(x['linear'][use], (choice[use] == y[use]).astype(int))
    return model, int(use.sum())


def query_scores(raw, y, groups, base):
    t, q = np.arange(len(y)), np.arange(len(y), len(raw))
    assert not np.isin(groups[t], groups[q]).any()
    neighbors = predict_neighbors(roster_tensor(groups), groups, y, t, q, 0)
    ordinary = [predictor(k, raw, y)(t, q, 0) for k in ['cat', 'budget']]
    return np.column_stack([base, *ordinary, *(neighbors[k] for k in NEIGHBOR)])


def export():
    raw, y, g, fold = data()
    test_path = ROOT/'data/raw/test.csv'
    test = pd.read_csv(test_path, usecols=lambda k: k != 'Name')
    all_raw = pd.concat([raw, test], ignore_index=True)
    groups = all_raw.PassengerId.str.split('_').str[0].to_numpy()
    np.testing.assert_array_equal(groups[:len(y)], g)
    parents = {}
    base, _, parents['v96'] = source('proposal_online_first_counts15', test)
    original, champion, parents['v131'] = source('proposal_v131_party_consensus', test)
    keep = champion != (base >= .5)
    assert keep.sum() == 61
    out = ROOT/'artifacts'/NAME
    report = ROOT/'reports'/NAME
    out.mkdir(exist_ok=True)
    gate, n = calibrator(raw, y, g, fold)
    model_file = out/'gate.joblib'
    joblib.dump(gate, model_file)
    scores = query_scores(all_raw, y, groups, base)
    fresh = query_scores(all_raw, y, groups, base)
    np.testing.assert_array_equal(scores, fresh)
    def decision(model, features):
        x, ref, choice = inputs(features, test, groups[len(y):])
        active = ~keep & (ref != choice)
        select = active & (model.predict_proba(x['linear'])[:, 1] >= .5)
        return np.where(select, features[:, -1], original), active, select
    p, active, change = decision(gate, scores)
    replay, _, _ = decision(joblib.load(model_file), fresh)
    np.testing.assert_array_equal(p, replay)
    np.testing.assert_array_equal((p >= .5) != champion, change)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all() and change.any()
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=p >= .5)).to_csv(out/'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=p, reload_probability=replay)).to_csv(
        out/'test_probabilities.csv', index=False)
    np.savez_compressed(out/'query_scores.npz', passenger=test.PassengerId.to_numpy(str), scores=scores)
    check = validate_submission(out/'submission.csv', test_path)
    distances = {}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        info = json.loads(receipt.read_text())
        assert check['sha256'] != info.get('file_sha256')
        if info.get('evaluation', {}).get('Status') != 'COMPLETE':
            continue
        old = ROOT/'artifacts'/receipt.parents[1].name/'submission.csv'
        assert sha256(old) == info['file_sha256']
        table = pd.read_csv(old)
        np.testing.assert_array_equal(table.PassengerId, test.PassengerId)
        distances[str(info['submission_ref'])] = int(((p >= .5) != table.Transported).sum())
    queued = ROOT/'artifacts/proposal_n23_service_pair_zero_donor/submission.csv'
    queued_pred = pd.read_csv(queued)
    np.testing.assert_array_equal(queued_pred.PassengerId, test.PassengerId)
    result = dict(**check, version=VERSION, model_sha256=sha256(model_file), reload_difference=0,
        parents=parents, training_rows=n, full_roster_rows=len(y), locked_v131=int(keep.sum()),
        eligible_test=int(active.sum()), v131_changes=int(change.sum()), changed_to_true=int((change & ~champion).sum()),
        changed_to_false=int((change & champion).sum()), receipts_checked=len(distances),
        closest_submitted_changes=min(distances.values()), different_from_queued=int(((p >= .5) != queued_pred.Transported).sum()),
        score_file_sha256=sha256(out/'query_scores.npz'))
    for directory in (out, report):
        write(directory/'manifest.json', result)
        write(directory/'selection.json', dict(version=VERSION, gate='pooled_half_roster_linear', threshold=.5))
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    export()
