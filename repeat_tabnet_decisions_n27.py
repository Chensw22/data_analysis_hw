"""Check whether the 23 independent TabNet decisions persist across booking calibration splits."""
import json
import numpy as np
import pandas as pd
from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from export_consensus_review_n26 import ART, ROOT, dig, write
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import cohort, count
from six_way_against_scored_nine_n26 import test_decisions

NAME = 'proposal_n27_independent_tabnet'
BASE = 'proposal_n27_directional_foundation'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/repeated_tabnet_correction.json'


def opposite(risk, first, cuts):
    return (risk[:, None] >= cuts[:, first.astype(int)].T).mean(1) > .5


def run(export=False):
    raw, y, groups, fold = data()
    _, blind, known, fresh = cohort(raw, y, fold)
    n = len(y)
    with np.load(ART / 'nov27_outsider_directional/aligned.npz') as z:
        net = z['tabnet'].mean(1)
        ids = z['ids']
        np.testing.assert_array_equal(z['y'], y)
        np.testing.assert_array_equal(z['f'], fold)
        original = z['TabNet_retained_random'][n:]
    with np.load(ART / 'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        first = z['first']
        np.testing.assert_array_equal(z['passenger'], ids)
    with np.load(ART / 'nov26_latent_passenger_difficulty/panel.npz') as z:
        core = np.median(z['scores'][:, -3:], axis=1)
        np.testing.assert_array_equal(z['passenger'], ids)
    risk = np.where(first, 1 - net, net)
    parent_risk = np.where(first, 1 - core, core)
    cuts = np.stack([calibrate(risk, first, y, np.flatnonzero(known & (fold == k)), 1.) for k in range(8)])
    current, parent_sha = test_decisions(BASE, pd.Series(ids[n:]))
    eligible = first[n:] == current
    test = eligible & opposite(risk[n:], first[n:], cuts)
    np.testing.assert_array_equal(test, original)
    trials = [partition(groups, 'independent_repeat_' + str(k), 4) for k in range(12)]
    trials += [splits(groups, fold, 'contiguous')]
    report = {'test_changes': int(test.sum()), 'splits': {}}
    votes = []
    parent = []
    for iteration, split in enumerate(trials):
        current_edits = np.zeros(n, bool)
        candidates = np.zeros(n, bool)
        for k in range(8):
            for j in range(4):
                t = np.flatnonzero(known & (fold == k) & (split != j))
                q = np.flatnonzero((fold == k) & (split == j))
                assert not np.isin(groups[t], groups[q]).any()
                for s, dest in [(risk,candidates), (parent_risk,current_edits)]:
                    cut = calibrate(s, first, y, t, 1.)
                    dest[q] = s[q] >= cut[first[q].astype(int)]
        votes.append(candidates)
        parent.append(current_edits)
        name = str(iteration) if iteration < 12 else 'contiguous'
        action = candidates & ~current_edits
        report['splits'][name] = {scope: count(action, mask, first[:n] != y) for scope, mask in
                                  [('all', known), ('fresh', known & fresh), ('unanimous', blind)]}
    a = (np.mean(votes[:12],axis=0) > .5) & ~(np.mean(parent[:12],axis=0) > .5)
    err = first[:n] != y
    d = a * np.where(err, 1, -1)
    report['majority'] = dict(all=count(a, known, err), fresh=count(a, fresh & known, err),
                              unanimous=count(a, blind, err), folds=[int(d[known & (fold == k)].sum()) for k in range(8)])
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(report, flush=True)
    if not export:
        return
    dest = ART / NAME
    dest.mkdir(exist_ok=True)
    parent_prob = pd.read_csv(ART / BASE / 'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(parent_prob.PassengerId, ids[n:])
    np.testing.assert_array_equal(parent_prob.probability >= .5, current)
    model = dest / 'inputs.npz'
    np.savez_compressed(model, passenger=ids[n:], baseline=first[n:], parent=parent_prob.probability.to_numpy(), net=net[n:], cuts=cuts)
    def apply(parent, first, net, cuts):
        act = (first == (parent >= .5)) & opposite(np.where(first, 1-net, net), first, cuts)
        return np.where(act, net, parent)
    scores = apply(parent_prob.probability.to_numpy(), first[n:], net[n:], cuts)
    with np.load(model) as z:
        replay = apply(z['parent'], z['baseline'], z['net'], z['cuts'])
    np.testing.assert_array_equal(scores, replay)
    np.testing.assert_array_equal(scores >= .5, current ^ test)
    pd.DataFrame({'PassengerId':ids[n:], 'Transported':scores >= .5}).to_csv(dest / 'submission.csv',index=False)
    pd.DataFrame({'PassengerId':ids[n:], 'probability':scores, 'reload_probability':replay}).to_csv(dest / 'test_probabilities.csv',index=False)
    checked = validate_submission(dest / 'submission.csv', ROOT / 'data/raw/test.csv')
    assert all(json.loads(p.read_text())['file_sha256'] != checked['sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest = dict(**checked, version=NAME.removeprefix('proposal_'), training_rows=n, parents={BASE:parent_sha},
                    reload_difference=0, model_sha256=dig(model), changed=int(test.sum()), source_file=__file__.split('/')[-1])
    for folder in (dest,ROOT/'reports'/NAME):
        write(folder/'manifest.json', manifest)
        write(folder/'selection.json', dict(version=manifest['version'], evidence=str(REPORT.relative_to(ROOT)),
              policy='Three TabNet outputs with side-specific thresholds; retain all directional foundation edits.'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    run(parser.parse_args().export)
