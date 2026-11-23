"""Separate local inversions from query-wide drift and allocations predictable from the incumbent."""
import numpy as np
from threadpoolctl import threadpool_limits

from broad_booking_neighborhood_proportions_n23 import OUT, REPORT
from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import SEEDS, write


def main():
    row, y, frame, groups, _ = load()
    _, _, hard = cohorts(row, y)
    size = np.minimum(frame.BookingSize.to_numpy(), 4)
    plans, history = {}, {}
    for seed in [*map(str,SEEDS), 'v131']:
        with np.load(OUT/f'{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            ref,base,fold=z['ref'],z['base'],z['fold']
            shift = z['residual_transfer_joint'] - base
        rules = {k: np.zeros(len(row)) for k in ('full','global','centered','risk_size_mean','local_excess')}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            adj = shift[q]
            medians = np.quantile(base[q], np.arange(1,10)/10)
            bins = np.searchsorted(medians,base[q])
            cell = size[q]*20 + 2*bins + ref[q]
            freq = np.bincount(cell)
            mean = (np.bincount(cell, weights=adj) / np.maximum(freq,1))[cell]
            for k,v in [('full',adj),('global',adj.mean()),('centered',adj-adj.mean()),
                        ('risk_size_mean',mean),('local_excess',adj-mean)]:
                rules[k][q]=base[q]+v
        before=ref==y
        hits={k:(np.where(ref==(base>=.5),v>=.5,ref)==y) for k,v in rules.items()}
        hits['reference']=before
        plans[seed]={k:paired(h,before) for k,h in hits.items()}
        if seed!='v131':
            for k,h in hits.items():history.setdefault(k,[]).append(h)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()})
    ref=np.stack(history['reference'])
    stable={}
    for k,s in history.items():
        h=np.stack(s)
        stable[k]=dict(hard_stable=int((h.all(0)&hard).sum()), gain95_pp=interval((h.astype(float)-ref).mean(0),groups))
    write(REPORT/'allocation.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(1):
        main()
