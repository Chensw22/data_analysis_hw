"""Compare equal label budgets allocated to whole bookings by uncertainty or lottery."""
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from exaone_validation_data import CATEGORICAL

REPORT = ROOT / 'reports/nov23_active_booking_labels'
OUT = ART / 'nov23_active_booking_labels'


def take(size, order, budget):
    selected = []
    remaining = budget
    for idx in order:
        if size[idx] <= remaining:
            selected.append(idx)
            remaining -= size[idx]
        if remaining == 0:
            break
    assert remaining == 0
    return np.array(selected)


def model(frame, y, t):
    fit = CatBoostClassifier(iterations=460, depth=6, learning_rate=.06, thread_count=2,
                             random_seed=1211, verbose=False, allow_writing_files=False)
    return fit.fit(frame.iloc[t], y[t], cat_features=CATEGORICAL)


def train(frame, y, group, t, q, seed):
    rng = np.random.default_rng(seed)
    keys, inv, count = np.unique(group[t], return_inverse=True, return_counts=True)
    initial = take(count, rng.permutation(len(keys)), len(t)//2)
    seen = np.isin(inv, initial)
    source, pool = t[seen], t[~seen]
    np.testing.assert_array_equal(np.intersect1d(group[pool], group[source]), [])
    small = model(frame, y, source)
    p = small.predict_proba(frame.iloc[pool])[:, 1].clip(1e-6, 1-1e-6)
    h = -p*np.log(p) - (1-p)*np.log1p(-p)
    keys, inv, size = np.unique(group[pool], return_inverse=True, return_counts=True)
    score = np.bincount(inv,weights=h)/size
    active = take(size,np.lexsort((rng.random(len(size)),-score)),len(t)//4)
    lottery = take(size,rng.permutation(len(keys)),len(t)//4)
    matched = np.concatenate([rng.choice(np.flatnonzero(size==s), np.sum(size[active]==s), replace=False)
                              for s in np.unique(size[active])])
    np.testing.assert_array_equal(np.sort(size[active]),np.sort(size[matched]))
    predictions = dict(initial=small.predict_proba(frame.iloc[q])[:,1])
    log = {}
    for key,chosen in [('uncertain',active),('lottery',lottery),('size_matched',matched)]:
        selected = np.isin(inv,chosen)
        new = pool[selected]
        assert len(new) == len(t)//4
        predictions[key] = model(frame,y,np.sort(np.r_[source,new])).predict_proba(frame.iloc[q])[:,1]
        log[key] = dict(passengers=len(new),bookings=len(chosen),mean_entropy=float(h[selected].mean()),
                        seed_mistake_rate=float(((p[selected]>=.5)!=y[new]).mean()))
    predictions['full'] = model(frame,y,t).predict_proba(frame.iloc[q])[:,1]
    return predictions, dict(initial=len(source), pool=len(pool), full=len(t), acquisitions=log)


def counts(h,b,m):
    return dict(correct=h[:,m].sum(1).tolist(), stable=int(h[:,m].all(0).sum()),
                fixed=(h[:,m]&~b[:,m]).sum(1).tolist(), harmed=(b[:,m]&~h[:,m]).sum(1).tolist())


def paired_interval(groups,active,control):
    _,ids=np.unique(groups,return_inverse=True)
    sizes=np.bincount(ids)
    difference=(active.astype(float)-control).mean(0)
    gains=np.bincount(ids,weights=difference)
    rng=np.random.default_rng(588)
    samples=[]
    for _ in range(40):
        sample=rng.integers(len(sizes),size=(50,len(sizes)))
        samples.extend(100*gains[sample].sum(1)/sizes[sample].sum(1))
    return dict(mean_pp=float(100*difference.mean()),ci95_pp=np.quantile(samples,[.025,.975]).tolist())


def main():
    row,y,frame,g,_ = load()
    OUT.mkdir(exist_ok=True)
    methods,refs,folds={},[],{}
    for seed in SEEDS:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as z:
            for k,v in [('y',y),('row',row)]:np.testing.assert_array_equal(z[k],v)
            base,fold=z['base'],z['fold']
        raw,logs={},[]
        for f in np.unique(fold):
            t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
            assert not np.intersect1d(g[t],g[q]).size
            scores,log=train(frame,y,g,t,q,seed+int(f))
            for k,p in scores.items():raw.setdefault(k,np.full(len(y),np.nan))[q]=p
            logs.append(log)
        for k,p in raw.items():
            assert np.isfinite(p).all()
            methods.setdefault(k,[]).append((p>=.5)==y)
        refs.append((base>=.5)==y)
        folds[str(seed)]=logs
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,base=base,fold=fold,**raw)
        print(seed,{k:round(float(h[-1].mean()),5) for k,h in methods.items()},flush=True)
    ids,labels,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    state=strata(frame)
    masks.update(paid=masks['repeat']&(state==2)&y,zero=masks['repeat']&(state==0)&~y)
    assert [masks[k].sum() for k in ['repeat','paid','zero']]==[1070,428,394]
    ref=np.stack(refs)
    masks['all']=np.ones(len(y),bool)
    grouped={k:summarize(np.stack(h),ref,masks) for k,h in methods.items()}
    direct={a+'_vs_'+b:{k:counts(np.stack(methods[a]),np.stack(methods[b]),m) for k,m in masks.items()}
            for a,b in [('uncertain','lottery'),('uncertain','size_matched'),('uncertain','initial'),
                        ('lottery','initial'),('full','initial')]}
    uncertainty={k:paired_interval(g,np.stack(methods['uncertain']),np.stack(methods[k]))
                 for k in ('lottery','size_matched','initial','full')}
    write(REPORT/'results.json',dict(plans=folds,grouped=grouped,comparisons=direct,uncertainty=uncertainty))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
