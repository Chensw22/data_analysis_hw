"""Test whether nested training rosters reproduce a neighborhood-removal decision."""
import json

import numpy as np

from full_roster_booking_validation_n23 import FAMILIES, OUT, ROOT, data
from full_roster_consensus_ranking_n23 import half_bank
from full_roster_metric_adjudication_n23 import load_scores
from full_roster_neighborhood_disputes_n23 import DEST

REPORT = ROOT / 'reports/nov23_matched_learning_curve/repeat_removal.json'


def outcomes(mask, flip, base, y):
    affected = mask & flip
    fixed = int(np.sum(affected & (base != y)))
    harmed = int(np.sum(affected & (base == y)))
    return dict(rows=int(mask.sum()), fixed=fixed, harmed=harmed, net=fixed-harmed)


def run():
    raw, y, groups, fold = data()
    panel = fold % 2
    for part in [fold, panel]:
        for f in np.unique(part):
            assert not np.isin(groups[part == f], groups[part != f]).any()
    full = np.column_stack([load_scores(raw, y, fold, OUT / f'{name}.npz', ['p'])[:, 0]
                            for name in (*FAMILIES, 'mitra96')])
    half = half_bank(raw, y, fold)
    base = full[:, -1] >= .5
    half_base = half[:, -1] >= .5
    candidates = []
    for name, split in [('full', fold), ('half', panel)]:
        scores = load_scores(raw, y, split, DEST / f'{name}.npz', ['without_near', 'control'])
        assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        candidates.append(scores >= .5)
    (propose, full_control), (corroborate, half_control) = [bank.T for bank in candidates]
    contrary = propose != base
    repeat = contrary & (corroborate == propose)
    # For a direct comparison, hold the set of proposed full-panel changes fixed
    # and replace only the corroborating half-panel model with its matched control.
    approved_by_control = contrary & (half_control == propose)
    rules = dict(full_proposal=contrary, repeat=repeat, repeat_against_half_mitra=repeat & (half_base != propose),
                 approved_by_half_control=approved_by_control,
                 repeated_control=(full_control != base) & (half_control == full_control),
                 contradicted_by_both_controls=repeat & (full_control != propose) & (half_control != propose))
    agree = ((full >= .5) == base[:, None]).all(1)
    rescued = contrary & agree & (propose == y)
    harmed = contrary & agree & (propose != y)
    assert rescued.sum() == 116
    masks = dict(all=np.ones(len(y), bool), unanimous=agree, divided=~agree,
                 unanimous_false=agree & ~base, unanimous_true=agree & base,
                 divided_false=~agree & ~base, divided_true=~agree & base,
                 full_proposal_shared_rescues=rescued, full_proposal_shared_harms=harmed)
    results = {}
    for name, flip in rules.items():
        by_group = {k: outcomes(mask, flip, base, y) for k, mask in masks.items()}
        by_group['fold_net'] = [outcomes(fold == f, flip, base, y)['net'] for f in np.unique(fold)]
        assert sum(by_group['fold_net']) == by_group['all']['net']
        results[name] = by_group
    # These discordant decisions capture the complete net difference between
    # using the half-panel neighborhood model and its training-size control.
    difference = {}
    for name, take in [('neighborhood_only', repeat & ~approved_by_control),
                       ('control_only', approved_by_control & ~repeat),
                       ('both', repeat & approved_by_control)]:
        difference[name] = {k: outcomes(mask, take, base, y) for k, mask in masks.items()}
    return dict(design='Both arms exclude the query order. The half training roster is nested in the '
                       'full training roster before removal. Each fits its own clustering. Repetition tests '
                       'their joint perturbation and does not constitute independent replication.',
                rows=len(y), baseline_correct=int((base == y).sum()), unanimous=int(agree.sum()),
                common_wrong=int((agree & (base != y)).sum()), full_proposal_shared_rescues=int(rescued.sum()),
                full_proposal_shared_harms=int(harmed.sum()), rules=results, corroboration_overlap=difference)


if __name__ == '__main__':
    result = run()
    REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, values in result['rules'].items():
        print(name, values['all'], values['unanimous'])
