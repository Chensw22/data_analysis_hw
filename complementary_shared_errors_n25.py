"""Challenge the unanimous archive across the complete passenger roster using six new feature views."""
import json
import numpy as np

from disjoint_tree_fields_n25 import ART, DEST, OUT, OUTER, RELEASE, stats
from full_roster_conditional_pool_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets


def votes(tag, rows, folds):
    p = np.empty((len(rows), 6))
    for f in np.unique(folds):
        for j, suffix in enumerate(('', '_repeat')):
            with np.load(OUT/f'{tag}_{f}{suffix}.npz') as z:
                np.testing.assert_array_equal(z['q'], rows[folds == f])
                p[folds == f, 3*j:3*j+3] = z['p']
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    return p >= .5


def compare(y, groups, folds, base, eligible, archive, challengers):
    # Evaluate using all rows with a known reference, even when all 68 stored models agree.
    agreed = (archive == base[:, None]).all(1)
    mistake = base != y
    dissent = challengers != base[:, None]
    report = dict(passengers=int(eligible.sum()), original_errors=int((eligible & mistake).sum()),
                  all_agreed_errors=int((eligible & mistake & agreed).sum()),
                  all_agreed_correct=int((eligible & ~mistake & agreed).sum()))
    for label, edit in [('any_view_opposes', dissent.any(1)),
                        ('both_majorities_oppose', (dissent[:, :3].sum(1) >= 2) & (dissent[:, 3:].sum(1) >= 2)),
                        ('all_six_oppose', dissent.all(1))]:
        delta = np.where(edit, np.where(mistake, 1, -1), 0)
        report[label] = dict(all=stats(delta, eligible, folds, groups),
                             unanimous=stats(delta, eligible & agreed, folds, groups))
    return report


def run():
    raw, y, groups, f = data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows, hf, hn = (z[k] for k in ('rows', 'fold', 'nine'))
    _, panels = prediction_sets(raw, y, groups, f, rows, hf)
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['fold'], f)
        en, ec = z['certified_nine'].copy(), z['certified_v131_equals_v96'].copy()
    for k in OUTER:
        with np.load(RELEASE/f'release{k}.npz') as z:
            q = z['q']
            np.testing.assert_array_equal(q, np.flatnonzero(f == k))
            en[q], ec[q] = z['nine'], z['v131'] >= 0
    scope = np.isin(f, OUTER)
    h = votes('historical', rows, hf)
    e = votes('full', np.arange(len(y)), f)
    historical, full = (np.column_stack(panel) >= .5 for panel in panels[::-1])
    result = {'historical': compare(y[rows], groups[rows], hf, hn, np.ones(len(rows), bool), historical, h)}
    for name, mask in [('three_folds', scope), ('remaining_five', ~scope),
                       ('previously_unseen_orders', ~np.isin(groups, groups[rows]))]:
        result[name] = compare(y, groups, f, en, ec & mask, full, e)
    DEST.with_name('complementary_shared_errors.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, r in result.items():
        print(name, r['original_errors'], r['all_agreed_errors'], [
            (k, r[k]['all']['fixed'], r[k]['all']['harmed'], r[k]['unanimous']['fixed'], r[k]['unanimous']['harmed'])
            for k in ('any_view_opposes', 'both_majorities_oppose', 'all_six_oppose')])


if __name__ == '__main__':
    run()
