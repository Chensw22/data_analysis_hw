"""Recheck the broad original-fold mistakes under repeated held-out bookings."""
from pathlib import Path

import numpy as np
import pandas as pd

from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent


def run():
    src=ROOT/'artifacts/nov23_global_error_inventory.csv'
    rows=pd.read_csv(src)
    y=rows.target.to_numpy(bool)
    wrong=rows.v131_wrong.to_numpy(bool)
    stubborn=wrong & rows.number_new_correct.eq(0).to_numpy()
    original_covered=wrong & ~stubborn
    hit=[]
    margin=[]
    for seed in [20261112,20261115,20261116]:
        with np.load(ROOT/f'artifacts/nov23_group_kernel/{seed}.npz') as z:
            np.testing.assert_array_equal(rows.row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            # Family pair plus two genuinely different standalone kernel models.
            # Requiring a rescue on every split stops a single split from dominating.
            probabilities=np.column_stack([z[k] for k in ['base','fixed','learned_twice']])
            hit.append(((probabilities>=.5)==y[:,None]).any(axis=1))
            margin.append(np.where(y,1-z['base'],z['base']))
    hits=np.sum(hit,axis=0)
    confidence=np.mean(margin,axis=0)
    result={}
    for label,mask in [('v131_errors',wrong),('previous_16_cover',original_covered),
                       ('previous_16_all_wrong',stubborn),('historical_and_16_all_wrong',stubborn&rows.historical_546)]:
        stats={str(i):int((mask&(hits==i)).sum()) for i in range(4)}
        result[label]=dict(rows=int(mask.sum()),seeds_with_a_correct_model=stats,
            still_wrong_false_negative=int((mask&(hits==0)&y).sum()),
            still_wrong_false_positive=int((mask&(hits==0)&~y).sum()),
            median_reference_wrong_class_probability=float(np.median(confidence[mask])))
    write(ROOT/'reports/global_error_models_nov23/grouped_coverage.json',result)
    print(result)


if __name__=='__main__':
    run()
