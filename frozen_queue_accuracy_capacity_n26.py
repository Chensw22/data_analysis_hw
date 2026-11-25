"""Bound each frozen submission's attainable total using only aggregate scores and decision counts."""
import json
from decimal import Decimal

import numpy as np
import pandas as pd

from audit_cross_family_submission_choices_n26 import inventory
from audit_flip_capacity_n23 import correct_count, required, same_denominator
from export_historical_veto_n26 import NAME as RESERVE
from forecast_frozen_queue_n26 import NAMES,ROOT
from six_way_against_scored_nine_n26 import test_decisions


def run():
    ids = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    n = len(ids)
    past = inventory(ids)
    digits = {int(Decimal(str(r['score']))*100000) for r in past.values()}
    assert same_denominator(digits,n,'either')==[n]
    threshold = required(n)
    parent = past[NAMES['nine']]
    starting = correct_count(str(parent['score']),n)
    report = dict(rows=n,scored_files=len(past),unique_scores=len(digits),goal_correct=threshold,
                  champion_correct=starting,candidates={})
    for key,name in list(dict(NAMES, reserve=RESERVE).items())[1:]:
        candidate,digest = test_decisions(name,ids)
        changes = int((candidate!=parent['vote']).sum())
        matches = []
        for reference,r in past.items():
            d = int(np.count_nonzero(candidate!=r['vote']))
            c = correct_count(str(r['score']),n)
            # The most optimistic case flips at most min(d, n-c) prior mistakes.
            upper = c+2*min(d,n-c)-d
            matches.append((upper,d,reference,c,r['ref']))
        upper,d,reference,c,receipt = min(matches)
        min_fixes = (threshold-starting+changes+1)//2
        report['candidates'][key] = dict(proposal=name,sha256=digest,changes_from_champion=changes,
                    maximum_correct=upper,maximum_accuracy=upper/n,ceiling_reference=reference,
                    ceiling_reference_ref=receipt,reference_correct=c,reference_distance=d,
                    target_possible_under_individual_bounds=upper>=threshold,
                    minimum_fixes_relative_to_champion=min_fixes if changes>=min_fixes else None,
                    required_precision_of_changes=min_fixes/changes if changes>=min_fixes else None)
        print(key,changes,round(100*upper/n,3),min_fixes if changes>=min_fixes else None,flush=True)
    report['interpretation'] = ('These are necessary constraints from each scored file separately, using only '
        'score totals and numbers of different predictions. An upper bound above the target is not an expected '
        'score or a proof that the goal is feasible. No hidden passenger labels are estimated.')
    (ROOT/'reports/nov23_matched_learning_curve/frozen_queue_accuracy_capacity.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
