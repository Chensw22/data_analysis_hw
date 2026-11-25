"""Quantify which registered passenger populations account for both releases' mistakes."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from expanded_joint_error_witnesses_n26 import load_scores
from outside_panel_coverage_n26 import archives, projection
from price_implied_deck_n26 import FEES
from target_mix_expert_routing_n26 import ROOT, sources


def run():
    raw, y, groups, folds, views, original = projection(include_full=True)
    nine, known, _, _ = champions()
    source = sources()[-1]['full']
    supplements = load_scores(raw, y, groups, {'full': source})['full']
    others, _ = archives(raw, y, groups, folds, True)
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    valid, parent = views['full']
    assert known[valid].all()
    same = valid & (parent == nine)
    wrong = parent != y
    primary_dissent = original != parent[:, None]
    primary_dissent[~paid, -2:] = False
    other_dissent = np.column_stack([v >= .5 for v in others.values()]) != parent[:, None]
    no_witness = ~(primary_dissent.any(1) | other_dissent.any(1) | ((supplements >= .5) != parent[:, None]).any(1))
    assert (same & wrong).sum() == 1232 and (same & wrong & no_witness).sum() == 178
    sleep = raw.CryoSleep.map({True: 'asleep', False: 'awake'}).fillna('unknown')
    active = np.where(paid, 'positive_spending', 'no_positive_spending')
    party = np.where(raw.groupby(groups).PassengerId.transform('size') == 1, 'alone', 'with_others')
    side = np.where(parent, 'predicted_transported', 'predicted_not_transported')
    deck = raw.Cabin.str.split('/').str[0].fillna('unknown')
    groupsets = dict(sleep_bill_direction=[sleep, active, side], order_size_direction=[party, side],
                     origin_deck=[raw.HomePlanet.fillna('unknown'), deck])
    report = {}
    for name, fields in groupsets.items():
        table = []
        labels = pd.DataFrame({i: v for i, v in enumerate(fields)})
        for key, rows in labels.groupby(list(labels.columns)).indices.items():
            ids = rows[same[rows]]
            if not len(ids):
                continue
            misses = ids[wrong[ids]]
            blind = ids[no_witness[ids]]
            table.append(dict(group=list(key), shared_predictions=len(ids), shared_errors=len(misses),
                              shared_error_rate=float(len(misses) / len(ids)), no_witness_rows=len(blind),
                              no_witness_errors=int(wrong[blind].sum()),
                              per_fold_errors=np.bincount(folds[misses], minlength=8).tolist()))
        assert sum(r['shared_errors'] for r in table) == 1232 and sum(r['no_witness_errors'] for r in table) == 178
        report[name] = sorted(table, key=lambda row: -row['shared_errors'])
        print(name, [(r['group'], r['shared_errors'], r['shared_predictions'], r['no_witness_errors'])
                     for r in report[name][:5]], flush=True)
    path = ROOT / 'reports/nov23_matched_learning_curve/shared_error_population.json'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    with threadpool_limits(1):
        run()
