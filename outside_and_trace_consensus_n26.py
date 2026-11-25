"""Pair a nested history-based reviewer with independent neural model families on the identical passengers."""
import json
import numpy as np
from nested_release_review_n26 import OUT,ROOT
from outside_panel_coverage_n26 import projection
from solo_archive_recovery_n26 import ticker


def run():
    _,y,g,fold,views,original=projection()
    report={}
    for tag in ('bookingmix','historical_to_new'):
        ids=np.flatnonzero(views[tag][0])
        parent=views[tag][1][ids]
        wrong=parent!=y[ids]
        hard=(original[ids]==parent[:,None]).all(1)
        def load(key):
            with np.load(OUT/f'{key}_{tag}.npz') as z:
                order=np.argsort(z['ids'])
                np.testing.assert_array_equal(z['ids'][order],ids)
                np.testing.assert_array_equal(z['wrong'][order],wrong)
                return {n:z[n][order] for n in z.files if n not in ('ids','wrong')}
        cells=load('cells')
        other=load('outsiders')
        hist=cells['both_new']
        ext=other['foundation_only']
        acts={'union_of_new_edits':hist|ext, 'both_new_edits':hist&ext,
              'history_foundation_support':hist&other['foundation'],
              'history_foundation_and_generative_support':hist&other['foundation']&other['generative'],
              'outside_addition_history_support':ext&cells['both']}
        r={}
        for n,act in acts.items():
            delta=act*np.where(wrong,1,-1)
            r[n]={k:ticker(mask,wrong,act) for k,mask in [('all',np.ones(len(ids),bool)),('all_99_agree',hard)]}
            r[n]['net_by_fold']=[int(delta[fold[ids]==k].sum()) for k in np.unique(fold[ids])]
        report[tag]=r
        print(tag,[(n,v['all']['fixed'],v['all']['harmed']) for n,v in r.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/outside_and_trace_consensus.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
