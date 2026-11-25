"""Audit origin evidence alone and paired across two training protocols."""
import json
import numpy as np

from disjoint_tree_fields_n25 import ART, DEST
from evaluate_inverse_tabicl_n26 import OUT, odds, ratio
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed


def decisions(rows, fold, paths):
    result = {name: np.zeros(len(rows), bool) for name in ('masked', 'observed', 'shuffled')}
    for k in np.unique(fold):
        mask = fold == k
        with np.load(paths(k)) as cache:
            np.testing.assert_array_equal(cache['q'], rows[mask])
            prior = odds(cache['prior'])
            assert np.isfinite(prior).all()
            result['masked'][mask] = prior >= 0
            for key in ('observed', 'shuffled'):
                signal = ratio(cache[key])
                assert np.isfinite(signal).all()
                result[key][mask] = prior + signal >= 0
    return result


def cohorts(view):
    agreement = (view['anchor'][:, None] == view['votes']).all(1)
    return {'all': view['known'], 'retained_agreement': view['eligible'] & agreement,
            'retained_disagreement': view['eligible'] & ~agreement,
            'previously_edited': view['known'] & ~view['eligible']}


def counts(mask, wrong, edit):
    fixed = int((mask & wrong & edit).sum())
    harmed = int((mask & ~wrong & edit).sum())
    return dict(rows=int(mask.sum()), errors=int((mask & wrong).sum()), fixed=fixed, harmed=harmed, net=fixed-harmed)


def run():
    raw, truth, groups, fold = data()
    _, _, _, rows, names, panel, old_panel, native, old_native, ref = panels_and_reference()
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        hf, old, parent = z['fold'], z['nine'], z['v131']
        np.testing.assert_array_equal(z['rows'], rows)
        np.testing.assert_array_equal(z['y'], truth[rows])
    awake = raw.CryoSleep.eq(False).to_numpy()
    hist = observed(rows, hf, old_panel, names, old_native, parent, old, old, awake, 'historical')
    full = observed(np.arange(len(raw)), fold, panel, names, native, *ref, awake, 'full')
    assert len(hist['cohort']) == 18
    hpred = decisions(rows, hf, lambda k: ART / f'nov26_historical_inverse_planet/{k}.npz')
    fpred = decisions(np.arange(len(raw)), fold, lambda k: OUT / f'{k}_home_planet.npz')
    hc, fc = cohorts(hist), cohorts(full)
    ha, fa = hist['anchor'], full['anchor'][rows]
    cold = np.zeros(len(rows), bool)
    for k in np.unique(hf):
        cold[hf == k] = ~np.isin(groups[rows[hf == k]], groups[rows[hf != k]])
    assert cold.sum() == 4691
    report = {}
    for scope, scope_mask in [('all', np.ones(len(rows), bool)), ('whole_order_held', cold)]:
        table = {}
        for cohort, eligible in hc.items():
            historical = scope_mask & eligible
            paired = historical & fc[cohort][rows] & (ha == fa)
            wrong = ha != truth[rows]
            for key in hpred:
                hist_edit = hpred[key] != ha
                full_edit = fpred[key][rows] != fa
                table.setdefault(cohort, {})[key] = {
                    'historical': counts(historical, wrong, hist_edit),
                    'paired_historical': counts(paired, wrong, hist_edit),
                    'paired_current': counts(paired, wrong, full_edit),
                    'repeated_edits': counts(paired, wrong, hist_edit & full_edit),
                    'per_historical_fold': [counts(historical & (hf == k), wrong, hist_edit) for k in np.unique(hf)]}
        report[scope] = table
    path = DEST.with_name('historical_inverse_planet.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    for name, table in report.items():
        for group in ('all', 'retained_agreement'):
            print(name, group, table[group]['observed']['historical'], table[group]['observed']['repeated_edits'])


if __name__ == '__main__':
    run()
