"""Release the frozen binary Mitra predictions while preserving the incumbent's earlier corrections."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_budget_quarter_n23 import source
from kaggle_workflow import sha256, validate_submission
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
BASE_NAME = 'proposal_n25_frozen_binary'
TREE = Path('artifacts/nov25_mitra_token_trees/predictions.npz')
STRATIFIED = Path('artifacts/nov25_mitra_stratified_support/test.npz')
NATIVE = 'proposal_online_first_counts15'
PREVIOUS = 'proposal_v131_party_consensus'
BEST = 'proposal_n24_selected_dissent'
CACHE = Path('artifacts/nov25_mitra_four_outcomes/test.npz')
REPORT = 'reports/nov23_matched_learning_curve/mitra_four_outcomes.json'


def apply(spec, train, test):
    path = ROOT / spec['component']
    assert sha256(path) == spec['component_sha256']
    native, hard, digest = source(NATIVE, test)
    _, previous, previous_digest = source(PREVIOUS, test)
    current, nine, best_digest = source(BEST, test)
    assert spec['parents'] == {NATIVE: digest, PREVIOUS: previous_digest, BEST: best_digest}
    with np.load(path) as z:
        np.testing.assert_array_equal(z['t'], np.arange(len(train)))
        np.testing.assert_array_equal(z['q'], np.arange(len(train), len(train) + len(test)))
        np.testing.assert_array_equal(z['passenger'], test.PassengerId)
        np.testing.assert_array_equal(z['native'], native.astype('float32'))
        np.testing.assert_array_equal(z['labels'], train.Transported.astype(int))
        np.testing.assert_array_equal(z['checkpoint'], spec['checkpoint'])
        scores = z['scores']
        alternative = scores[:, 0]
    assert np.isfinite(alternative).all() and ((alternative >= 0) & (alternative <= 1)).all()
    eligible = (hard == previous) & (previous == nine)
    if spec['policy'] == 'binary_confirmed':
        from frozen_joint_corroboration_n25 import decisions
        assert sha256(ROOT / TREE) == spec['tree_sha256']
        with np.load(ROOT / TREE) as z:
            np.testing.assert_array_equal(z['test_ids'], test.PassengerId)
            tree = z['test_by_fold'].mean(axis=0)[:, z['model_names'].tolist().index('joint')]
        eligible = decisions(scores, tree, native, previous, nine)['binary_confirmed']
    if spec['policy'] == 'matched_0.5':
        from mitra_four_outcomes_n25 import FEES
        assert sha256(ROOT / STRATIFIED) == spec['stratified_sha256']
        with np.load(ROOT / STRATIFIED) as z:
            np.testing.assert_array_equal(z['q'], np.arange(len(train), len(train) + len(test)))
            np.testing.assert_array_equal(z['t'], np.arange(len(train)))
            np.testing.assert_array_equal(z['passenger'], test.PassengerId)
            np.testing.assert_array_equal(z['active'], test[FEES].gt(0).any(axis=1))
            np.testing.assert_array_equal(z['checkpoint'], spec['checkpoint'])
            alternative = .5 * alternative + .5 * z['scores'][:, 0]
    return np.where(eligible, alternative, current), nine


def run(confirmed, stratified=False):
    name = BASE_NAME + ('_confirmed' if confirmed else '')
    policy = 'binary_confirmed' if confirmed else 'binary_1.0'
    report = 'reports/nov23_matched_learning_curve/frozen_joint_corroboration.json' if confirmed else REPORT
    if stratified:
        name = BASE_NAME + '_matched_support'
        policy = 'matched_0.5'
        report = 'reports/nov23_matched_learning_curve/mitra_stratified_support.json'
    train = pd.read_csv(ROOT / 'data/raw/train.csv', usecols=['Transported'])
    test_file = ROOT / 'data/raw/test.csv'
    test = pd.read_csv(test_file, usecols=lambda c: c != 'Name')
    parents = {name: source(name, test)[2] for name in (NATIVE, PREVIOUS, BEST)}
    with np.load(ROOT / CACHE) as z:
        checkpoint = str(z['checkpoint'])
    for fold in range(3):
        with np.load(ROOT / CACHE.parent / f'fold{fold}.npz') as z:
            np.testing.assert_array_equal(z['checkpoint'], checkpoint)
    spec = dict(policy=policy, parents=parents, checkpoint=checkpoint, component=str(CACHE), component_sha256=sha256(ROOT / CACHE),
                rule='Use frozen binary scores unless v131 or the nine-vote release changed the original decision.')
    if confirmed:
        spec['tree_sha256'] = sha256(ROOT / TREE)
        spec['rule'] += ' Require corroboration from both spending outputs or the token tree.'
    if stratified:
        spec['stratified_sha256'] = sha256(ROOT / STRATIFIED)
        spec['rule'] += ' Average the full-context probability with the spending-matched context probability.'
    folder = ROOT / 'artifacts' / name
    folder.mkdir(exist_ok=True)
    composition = folder / 'composition.json'
    write(composition, spec)
    scores, baseline = apply(spec, train, test)
    reloaded, _ = apply(json.loads(composition.read_text()), train, test)
    np.testing.assert_array_equal(scores, reloaded)
    hard = scores >= .5
    changed = hard != baseline
    pd.DataFrame(dict(PassengerId=test.PassengerId, Transported=hard)).to_csv(folder / 'submission.csv', index=False)
    pd.DataFrame(dict(PassengerId=test.PassengerId, probability=scores, reload_probability=reloaded)).to_csv(
        folder / 'test_probabilities.csv', index=False)
    check = validate_submission(folder / 'submission.csv', test_file)
    distances = []
    for receipt in (ROOT / 'reports').glob('*/online_attempts/*.json'):
        record = json.loads(receipt.read_text())
        assert record.get('file_sha256') != check['sha256'], receipt
        other = receipt.parent.parent.name
        submitted = ROOT / 'artifacts' / other / 'submission.csv'
        if submitted.is_file() and sha256(submitted) == record.get('file_sha256'):
            old = pd.read_csv(submitted)
            np.testing.assert_array_equal(old.PassengerId, test.PassengerId)
            distances.append((int((old.Transported.to_numpy() != hard).sum()), other))
    closest_distance, closest_name = min(distances)
    assert closest_distance > 0 and changed.any()
    manifest = dict(**check, version=name.removeprefix('proposal_'), reload_difference=0, parents=parents,
                    model_sha256=sha256(composition), training_rows=len(train), edits_vs_best=int(changed.sum()),
                    edits_to_true=int((changed & hard).sum()), validation=report,
                    compared_archived_exports=len(distances), nearest_scored_export=closest_name,
                    differences_from_nearest=closest_distance)
    for dest in (folder, ROOT / 'reports' / name):
        write(dest / 'manifest.json', manifest)
        write(dest / 'selection.json', dict(version=manifest['version'], threshold=.5, rule=spec['rule'],
                                           validation=report, policy=policy))
    print(manifest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--confirmed', action='store_true')
    group.add_argument('--stratified', action='store_true')
    args = parser.parse_args()
    run(args.confirmed, args.stratified)
