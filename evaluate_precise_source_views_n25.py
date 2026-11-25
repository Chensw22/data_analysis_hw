"""Compare exact-amount and binned manifest views, tracking the same omitted source across models."""
import argparse
import json
import numpy as np

from disjoint_tree_fields_n25 import ART,DEST,stats
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF
from precise_source_views_n25 import OUT,GROUPS


def flips(ref,tab,gen,tab_full,gen_full):
    a=(tab>=.5)!=ref[:,None]
    b=(gen>=.5)!=ref[:,None]
    both=a&b
    changed_by_omission=((tab_full>=.5)==ref)&((gen_full>=.5)==ref)
    d={f'raw_omit_{k}':a[:,j] for j,k in enumerate(GROUPS)}
    d.update({f'same_omit_{k}':both[:,j] for j,k in enumerate(GROUPS)})
    d.update(raw_any=a.any(1),same_source_any=both.any(1),same_source_twice=both.sum(1)>=2,
             same_source_both_changed=changed_by_omission&both.any(1),
             mismatched_sources_any=(a&np.roll(b,1,axis=1)).any(1),
             raw_majority=a.sum(1)>2,raw_full=(tab_full>=.5)!=ref)
    return d


def run(folds):
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    majority=votes>34
    unanimous=(votes==0)|(votes==68)
    near=(votes<=3)|(votes>=65)
    blocks={}
    masks=[]
    opportunities=[]
    for k in folds:
        q=np.flatnonzero(f==k)
        with np.load(REF/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            low,high,nine=(z[n] for n in ('nine_on_low_branch','nine_on_high_branch','nine'))
        masks.append((q,nine>=0))
        with np.load(ART/f'nov25_passenger_joint_likelihood/fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            gen=np.column_stack([z['omit_'+s] for s in GROUPS])
            gen_full=z['direct']
            broad=np.column_stack([z[n] for n in z.files if n in ('direct','outcome_only','chain_0','chain_1') or n.startswith(('restore_','omit_'))])
        with np.load(ART/f'nov25_passenger_state_contexts/{k}_global.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            tab_full=z['p']
        tab=[]
        for name in GROUPS:
            with np.load(OUT/f'{k}_{name}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                tab.append(z['p'])
        tab=np.column_stack(tab)
        alternatives=[flips(ref,tab,gen,tab_full,gen_full) for ref in (low,high,majority[q])]
        for name in alternatives[0]:
            triple=[]
            for ref,choice in zip((low,high,majority[q]),alternatives,strict=True):
                triple.append(((ref^choice[name])==y[q]).astype(int)-(ref==y[q]))
            blocks.setdefault(name,[]).append(triple)
        gen_hit=((broad>=.5)!=majority[q,None]).any(1)
        raw_hit=alternatives[2]['raw_any']
        opportunities.append(np.column_stack([gen_hit,raw_hit,gen_hit&raw_hit,gen_hit|raw_hit,
                                              alternatives[2]['same_source_any']]))
    q=np.concatenate([x for x,_ in masks])
    known=np.concatenate([m for _,m in masks])
    same=unanimous[q]
    wrong=majority[q]!=y[q]
    all_rows=np.ones(len(q),bool)
    opportunity={}
    for name,mask in zip(('binned_all','raw_omissions','both_any_view','either_any_view','same_omission'),
                         np.vstack(opportunities).T,strict=True):
        opportunity[name]=dict(recovered=int((wrong&same&mask).sum()),exposed=int((~wrong&same&mask).sum()))
    result={}
    for name,parts in blocks.items():
        a,b,arc=(np.concatenate([v[j] for v in parts]) for j in range(3))
        np.testing.assert_array_equal(a[known],b[known])
        def calc(d,m):return stats(d,m,f[q],g[q])
        result[name]=dict(all_bounds=[int(np.minimum(a,b).sum()),int(np.maximum(a,b).sum())],
                          nine=calc(a,known),new_orders=calc(a,known&~np.isin(g[q],g[rows])),
                          all_archive=calc(arc,all_rows),near=calc(arc,near[q]),unanimous=calc(arc,same))
        r=result[name]
        print(name,r['all_bounds'],r['nine']['fixed'],r['nine']['harmed'],
              r['unanimous']['fixed'],r['unanimous']['harmed'],flush=True)
    print(opportunity)
    target=DEST.with_name('precise_source_views.json') if len(folds)==8 else OUT/'interim.json'
    target.write_text(json.dumps(dict(folds=folds,rows=len(q),known=int(known.sum()),
        unanimous_errors=int((wrong&same).sum()),opportunity=opportunity,rules=result),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folds',nargs='+',type=int,choices=range(8),default=list(range(8)))
    run(p.parse_args().folds)
