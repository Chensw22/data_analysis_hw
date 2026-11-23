"""Replay the distillation recipe at two training sizes on identical unseen bookings."""
import hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import load_roster
from broad_crossfamily_unlabelled_distillation_n23 import predict, student_frame
from broad_error_learning_curve_n23 import paired_changes
from broad_partition_evidence_n23 import ART,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from matched_booking_learning_curve_n23 import OUT as OLD, portions
from tabicl_routed_context_n23 import DIGEST,WEIGHT


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,masks=historical()
    compact,g,size,coverage=load_roster()
    external=np.arange(len(compact)-coverage['test_rows'],len(compact))
    assert not np.intersect1d(g[external],g[row]).size
    frame=student_frame(compact,size)
    state=strata(frame.iloc[row])
    scopes=dict(all=np.ones(len(y),bool),**masks,paid_transported=masks['repeat']&(state==2)&y,
                zero_stayed=masks['repeat']&(state==0)&~y)
    scopes['core']=scopes['paid_transported'] | scopes['zero_stayed']
    scopes['other']=~scopes['core']
    out=ART/'nov23_near_full_distillation'
    out.mkdir(exist_ok=True)
    bank={}
    for family in ['v44','tabicl']:
        with np.load(OLD/f'{family}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            if bank:
                np.testing.assert_array_equal(z['fold'],fold)
                np.testing.assert_array_equal(z['sizes'],sizes)
            fold,sizes=z['fold'],z['sizes']
            bank[family]=np.stack([z['small'],z['large']])
    bank['reference']=(bank['v44']+bank['tabicl'])/2
    for k,s,t,q in portions(g[row]):
        np.testing.assert_array_equal(np.flatnonzero(fold==k),q)
        np.testing.assert_array_equal(sizes[k],[len(s),len(t),len(q)])
        assert np.isin(s,t).all() and not np.intersect1d(g[row[t]],g[row[q]]).size
        for j,train in enumerate([s,t]):
            path=out/f'fold_{k}_size_{j}.npz'
            if path.exists():
                with np.load(path) as cache:
                    np.testing.assert_array_equal(cache['training_rows'],row[train])
                    np.testing.assert_array_equal(cache['query_rows'],row[q])
                    np.testing.assert_array_equal(cache['training_labels'],y[train])
                    raw={name:cache[name] for name in ['teacher','native','external','target_mix','shuffled']}
            else:
                raw=predict(compact,frame,row[train],row[q],external,y[train],k)
                pending=path.with_suffix('.partial.npz')
                np.savez_compressed(pending,training_rows=row[train],training_labels=y[train],query_rows=row[q],**raw)
                pending.replace(path)
            for name,p in raw.items():bank.setdefault(name,np.full((2,len(y)),np.nan))[j,q]=p
        print('completed matched fold',k+1,flush=True)
    for name in ['native','target_mix','shuffled','external']:
        bank['ensemble_'+name]=(2*bank['reference']+bank[name])/3
    metrics,by_size={},{}
    groups=g[row]
    for k,probs in bank.items():
        assert np.isfinite(probs).all()
        hits=(probs>=.5)==y[None,:]
        metrics[k]=dict(acc=hits.mean(1).tolist(),coverage={name:hits[:,s].sum(1).tolist() for name,s in scopes.items()},
                        added_training=paired_changes(hits[:1],hits[1:],groups,scopes))
    for j,label in enumerate(['small','large']):
        comparisons={}
        for name,control in [('target_mix','native'),('target_mix','shuffled'),('ensemble_target_mix','ensemble_native'),
                             ('ensemble_target_mix','reference'),('external','native')]:
            h=(bank[name][j]>=.5)==y
            old=(bank[control][j]>=.5)==y
            comparisons[name+'_vs_'+control]=dict(fixed=int((h&~old).sum()),harmed=int((~h&old).sum()),
                change=paired_changes(old[None,:],h[None,:],groups,scopes))
        by_size[label]=comparisons
    np.savez_compressed(out/'predictions.npz',row=row,y=y,fold=fold,sizes=sizes,**bank)
    write(REPORT/'near_full.json',dict(models=metrics,matched=by_size,sizes=sizes.tolist()))
    print('reference / student / ensemble',*[metrics[k]['acc'] for k in ['reference','target_mix','ensemble_target_mix']],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
