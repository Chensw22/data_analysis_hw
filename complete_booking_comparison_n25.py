"""Audit the frozen 30-edit release and shared errors after reconstructing all eight booking folds."""
import json
import numpy as np

from disjoint_tree_fields_n25 import ART, DEST, OUT as VIEWS, stats
from full_roster_booking_validation_n23 import data
from full_roster_permutation_selection_n24 import OUT as TREES
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from grouped_peer_teacher_n25 import OUT


def run():
    raw,y,g,fold=data()
    rows_all=np.arange(len(y))
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,fold,rows,hf)
    counts=(np.column_stack(panels[0])>=.5).sum(1)
    correct_votes=np.where(y,counts,68-counts)
    consensus=(counts==0)|(counts==68)
    common=correct_votes==0
    same_order=raw.PassengerId.str.split('_').str[0]
    solo=~same_order.duplicated(keep=False).to_numpy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    new_order=~np.isin(g,g[rows])
    native=np.empty(len(y))
    parent,nine=np.full(len(y),-1),np.full(len(y),-1)
    low,high,random,confirmed=(np.zeros(len(y),bool) for _ in range(4))
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        with np.load(OUT/f'release{k}.npz') as z:
            for key,val in [('q',q),('ids',raw.PassengerId.iloc[q]),('y',y[q]),('party',~solo[q])]:
                np.testing.assert_array_equal(z[key],val)
            native[q],parent[q],nine[q],low[q],high[q]=(z[s] for s in ('native','v131','nine','nine_on_low_branch','nine_on_high_branch'))
        with np.load(TREES/f'full_{k}.npz') as z:
            np.testing.assert_array_equal(z['query'],q)
            mix=.75*native[q]+.25*z['prob'][:,2]
        random[q]=mix>=.5
        endorsed=np.ones(len(q),bool)
        for suffix in ('','_repeat'):
            with np.load(VIEWS/f'full_{k}{suffix}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                endorsed &= ((z['p']>=.5)==random[q,None]).sum(1)>=2
        confirmed[q]=np.where(endorsed,random[q],native[q]>=.5)
    known=parent>=0
    assert ((nine[known]==low[known]) & (nine[known]==high[known])).all()
    report={'rows':len(y),'resolved_nine':int((nine>=0).sum()),'resolved_parent':int(known.sum()),
            'resolved_party':int((known&~solo).sum()),'total_party':int((~solo).sum()),
            'full_nine_correct_bounds':[int(((low==y)&(high==y)).sum()),int(((low==y)|(high==y)).sum())],
            'correct_required_to_exceed_85':int(.85*len(y))+1,
            'consensus':dict(unanimous=int(consensus.sum()),wrong=int(common.sum()),
                             wrong_with_known_nine=int((common&(nine>=0)).sum()),
                             wrong_with_known_nine_also_wrong=int((common&(nine>=0)&(nine!=y)).sum()),
                             wrong_within_three=int((correct_votes<=3).sum()), known_awake_positive=int((common&awake&y).sum()),
                             awake_solo_positive=int((common&awake&solo&y).sum()), complete_fields=int((common&raw.notna().all(axis=1)).sum()),
                             new_order=int((common&new_order).sum()))}
    for name,pred in [('original_79',random),('confirmed_30',confirmed)]:
        a,b=edit_effect_bounds(pred,native>=.5,parent,low,high,y)
        assert np.array_equal(a[known],b[known])
        result={}
        for section,mask in [('all',rows_all>=0),('first_three',fold<3),('remaining_five',fold>=3),
                             ('solo',solo),('party',~solo),('new_orders',new_order),
                             ('unanimous',consensus&((counts>34)==nine))]:
            result[section]=dict(bounds=[int(a[mask].sum()),int(b[mask].sum())],
                                 exact_reference=stats(a,mask&known,fold,g))
        report[name]=result
    DEST.with_name('complete_booking_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for k in ('resolved_nine','full_nine_correct_bounds','consensus'):
        print(k,report[k])
    for key in ('original_79','confirmed_30'):
        print(key, [(n, v['bounds'], v['exact_reference']['fixed'],v['exact_reference']['harmed'])
                    for n,v in report[key].items()])


if __name__=='__main__':
    run()
