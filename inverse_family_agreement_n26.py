"""Measure whether inverse attribute predictions add corrections outside 18 families' reach."""
import json

import numpy as np

from disjoint_tree_fields_n25 import ART, DEST
from evaluate_inverse_tabicl_n26 import OUT, VIEWS, odds, ratio
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from outstanding_error_families_n26 import observed


def run():
    raw, truth, groups, folds = data()
    _, _, _, history, names, panel, _, native, _, reference = panels_and_reference()
    awake = raw.CryoSleep.eq(False).to_numpy()
    view = observed(np.arange(len(raw)), folds, panel, names, native, *reference, awake, 'full')
    anchor = view['anchor']
    wrong = anchor != truth
    agree = (view['votes'] == anchor[:, None]).all(axis=1)
    strata = {
        'all': view['known'],
        'retained_agreement': view['eligible'] & agree,
        'retained_disagreement': view['eligible'] & ~agree,
        'previously_edited': view['known'] & ~view['eligible'],
    }
    np.testing.assert_array_equal(sum(s.astype(int) for s in list(strata.values())[1:]), view['known'])
    assert (wrong & strata['retained_agreement']).sum() == 607
    votes = {source: np.zeros((len(raw), len(VIEWS)), bool) for source in ('masked', 'observed', 'shuffled')}
    forward = np.zeros(len(raw), bool)
    for fold in np.unique(folds):
        held = np.flatnonzero(folds == fold)
        with np.load(ART / f'nov25_passenger_state_contexts/{fold}_global.npz') as cache:
            np.testing.assert_array_equal(cache['q'], held)
            forward[held] = cache['p'] >= .5
        for column, name in enumerate(VIEWS):
            with np.load(OUT / f'{fold}_{name}.npz') as cache:
                np.testing.assert_array_equal(cache['q'], held)
                prior = odds(cache['prior'])
                likelihood = {source: ratio(cache[source]) for source in ('observed', 'shuffled')}
            if name == 'dominant_service':
                with np.load(OUT / f'{fold}_paid_identity.npz') as paid:
                    eligible = np.isin(held, paid['q'])
                for values in likelihood.values():
                    values[~eligible] = 0
            votes['masked'][held, column] = prior >= 0
            for source, values in likelihood.items():
                votes[source][held, column] = prior + values >= 0
    edits = {'forward': forward != anchor}
    for source, labels in votes.items():
        disagreement = labels != anchor[:, None]
        for column, name in enumerate(VIEWS):
            edits[f'{source}_{name}'] = disagreement[:, column]
        opponents = disagreement.sum(axis=1)
        edits[source + '_any'] = opponents >= 1
        edits[source + '_majority'] = opponents >= 2
        edits[source + '_all'] = opponents == len(VIEWS)
    new = ~np.isin(groups, groups[history])
    records = {}
    for partition, subset in [('all', np.ones(len(raw), bool)), ('new_orders', new)]:
        records[partition] = {}
        for name, cohort in strata.items():
            eligible = cohort & subset
            counts = {}
            for method, change in edits.items():
                flagged = change & eligible
                counts[method] = dict(fixed=int((flagged & wrong).sum()), harmed=int((flagged & ~wrong).sum()))
            records[partition][name] = dict(rows=int(eligible.sum()), errors=int((eligible & wrong).sum()), methods=counts)
    path = DEST.with_name('inverse_family_agreement.json')
    path.write_text(json.dumps(records, indent=2) + '\n')
    for partition, values in records.items():
        print(partition, values['retained_agreement'])


if __name__ == '__main__':
    run()
