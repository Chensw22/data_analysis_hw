"""Challenge the v131 release with an untrained rank-normalized multi-family center."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata

from broad_booking_policy_replay_n23 import interval
from broad_pairwise_adjudication_n23 import paired
from error_overlap_diagnostic import FAMILIES, load_authenticated
from export_consensus_routes import BASE, HISTORICAL, official_decisions
from export_nov21_error_batch import test_probabilities
from kaggle_workflow import sha256, validate_submission
from released_correction_audit_n23 import ART, ROOT, development
from relational_error_corrector import write

OUT = ART/'nov23_robust_release_center'
REPORT = ROOT/'reports/nov23_robust_release_center'


def decisions(bank, anchor, incumbent, batches):
    """Scale margins against the unlabeled query batch and reduce extreme model opinions."""
    keys = ['soft_mean', 'raw_trim', 'rank_trim']
    results = {k: incumbent.copy() for k in keys}
    for i in np.unique(batches):
        rows = np.flatnonzero(batches == i)
        p = bank[rows]
        signed = 2*(p >= .5).astype(float)-1
        strength = np.column_stack([rankdata(np.abs(p[:, j]-.5), method='average')/(len(p)+1)
                                    for j in range(p.shape[1])])
        ranks = np.sort(signed * strength, axis=1)
        raw = np.sort(logit(p.clip(.001, .999)), axis=1)
        scores = {'soft_mean': p.mean(axis=1)-.5,
                  'raw_trim': raw[:, 1:-1].mean(axis=1),
                  'rank_trim': ranks[:, 1:-1].mean(axis=1)}
        for k, s in scores.items():
            locked = incumbent[rows] != (anchor[rows] >= .5)
            results[k][rows] = np.where(locked | (s == 0), incumbent[rows], s > 0)
    return results


def comparison(hit, old, folds, cold, groups):
    gain = hit.astype(int)-old.astype(int)
    return dict(**paired(hit, old), group_gain95_pp=interval(gain, groups),
                net_by_fold=[int(gain[folds==i].sum()) for i in np.unique(folds)],
                cold=paired(hit[cold], old[cold]), cold_gain95_pp=interval(gain[cold], groups[cold]),
                cold_net_by_fold=[int(gain[(folds == i) & cold].sum()) for i in np.unique(folds)])


def main():
    inp, splits, folds, model, _ = load_authenticated()
    y, releases, masks = development()
    np.testing.assert_array_equal(inp['y'], y)
    group = np.array(inp['groups'])
    assert len(group)==len(y)
    cold = np.zeros(len(y), dtype=bool)
    for t, q in splits:
        cold[q] = ~np.isin(group[q], group[t])
    assert cold.sum()==4691
    with np.load(ART/'v108_exact_oof/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'], inp['dev'])
        np.testing.assert_array_equal(z['fold'], folds)
        np.testing.assert_array_equal(z['true'], y)
        anchor = z['v108']
    bank = np.column_stack([model[k] for k in FAMILIES] + [anchor])
    local = decisions(bank, anchor, releases[131], folds)
    ids = pd.read_csv(ROOT/'data/raw/test.csv', dtype={'PassengerId': str}).PassengerId
    p = test_probabilities(BASE, ids)
    old, _ = official_decisions('proposal_v131_party_consensus', ids)
    official_bank = np.column_stack([test_probabilities(f, ids) for f in HISTORICAL] + [p])
    final = decisions(official_bank, p, old, np.zeros(len(p), int))
    OUT.mkdir(exist_ok=True)
    verified = {}
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        receipt = json.loads(path.read_text())
        if (receipt.get('evaluation') or {}).get('Status') != 'COMPLETE':
            continue
        file = ART/path.parent.parent.name/'submission.csv'
        if file.is_file() and sha256(file) == receipt.get('file_sha256'):
            frame = pd.read_csv(file, dtype={'PassengerId': str})
            np.testing.assert_array_equal(ids, frame.PassengerId)
            verified[path.parent.parent.name] = frame.Transported.to_numpy()
    assert 'proposal_v131_party_consensus' in verified
    stats = {}
    for k, vote in local.items():
        h, b = vote == y, releases[131] == y
        distances = sorted((int((final[k] != v).sum()), name) for name,v in verified.items())
        stats[k] = dict(vs_champion=comparison(h,b,folds,cold,group),
                        vs_v116=comparison(h,releases[116]==y,folds,cold,group),
                        historical_errors={name: paired(h[m], b[m]) for name,m in masks.items()},
                        official_test_changes=int((final[k]!=old).sum()), nearest_submitted=distances[:3])
    # Export the declared rule; the decision to send is made separately on the evidence.
    key='rank_trim'
    submission = OUT/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids, Transported=final[key])).to_csv(submission, index=False)
    check = validate_submission(submission, ROOT/'data/raw/test.csv')
    write(OUT/'selection.json', dict(version='confidence_rank_trim_six_v131_locked', **check))
    write(REPORT/'selection.json', dict(version='confidence_rank_trim_six_v131_locked', **check))
    write(REPORT/'result.json', dict(families=[*FAMILIES,'v108'], development_rows=len(y),
          local_folds=len(splits), cold_rows=int(cold.sum()), verified_archive=len(verified),
          retained_v131_development_overrides=int((releases[131]!=(anchor>=.5)).sum()),
          retained_v131_test_overrides=int((old!=(p>=.5)).sum()),
          results=stats, candidate=str(submission.relative_to(ROOT)), sha256=sha256(submission)))
    print(json.dumps(stats[key]['vs_champion']),stats[key]['nearest_submitted'],flush=True)


if __name__=='__main__':
    main()
