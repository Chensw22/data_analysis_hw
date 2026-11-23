"""Separate the effects of more true labels and distillation on the same held-out passengers."""
import numpy as np

from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT


def paired_intervals(delta,group,scopes):
    _,code=np.unique(group,return_inverse=True)
    cnt=np.column_stack([np.bincount(code,weights=m) for m in scopes.values()])
    change=np.stack([np.column_stack([np.bincount(code,weights=d*m) for m in scopes.values()])
                     for d in delta.values()])
    sums=change.sum(axis=1)
    draws=[]
    rng=np.random.default_rng(521)
    for _ in range(50):
        take=rng.integers(len(cnt),size=(40,len(cnt)))
        draws.append(100*change[:,take,:].sum(2)/cnt[take].sum(1)[None,:,:])
    lo,hi=np.quantile(np.concatenate(draws,axis=1),[.025,.975],axis=1)
    return {k:{name:dict(net=float(sums[i,j]),change_pp=float(100*sums[i,j]/cnt[:,j].sum()),
                         interval95_pp=[float(lo[i,j]),float(hi[i,j])])
               for j,name in enumerate(scopes)} for i,k in enumerate(delta)}


def main():
    row,y,frame,group,_=load()
    ids,truth,m=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,truth)
    same_wrong=np.ones(len(y),bool)
    for seed in SEEDS:
        with np.load(ART/f'nov23_crossfamily_unlabelled_distillation/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            same_wrong &= ((z['teacher']>=.5)!=y)&((z['native']>=.5)!=y)
    assert same_wrong.sum()==818 and (same_wrong&m['repeat']).sum()==791
    state=strata(frame)
    scopes=dict(all=np.ones(len(y),bool),ever=m['ever'],repeat=m['repeat'],parents_wrong=same_wrong,
                paid_transported=m['repeat']&y&(state==2),zero_stayed=m['repeat']&~y&(state==0))
    with np.load(ART/'nov23_near_full_distillation/predictions.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        hit={k:((z[k]>=.5)==y[None,:]).astype(float) for k in ['native','teacher','target_mix','reference','ensemble_native','ensemble_target_mix']}
    output={}
    for kind,ordinary,distilled in [('standalone','native','target_mix'),('ensemble','ensemble_native','ensemble_target_mix')]:
        ns,nl=hit[ordinary]
        ds,dl=hit[distilled]
        # Both orders contribute equally to the allocation of the total net difference.
        delta=dict(total=dl-ns,training_size=.5*((nl-ns)+(dl-ds)),recipe=.5*((ds-ns)+(dl-nl)),
                   interaction=(dl-ds)-(nl-ns))
        np.testing.assert_allclose(delta['total'],delta['training_size']+delta['recipe'])
        output[kind]=paired_intervals(delta,group,scopes)
    detail={}
    for k,h in hit.items():
        detail[k]={name:dict(small=int(h[0,mask].sum()),large=int(h[1,mask].sum()),
                             fixed=int(((h[1]>h[0])&mask).sum()),harmed=int(((h[1]<h[0])&mask).sum()))
                  for name,mask in scopes.items()}
    write(REPORT/'near_full_attribution.json',dict(effects=output,coverage=detail,counts={k:int(v.sum()) for k,v in scopes.items()}))
    print(output['standalone']['training_size']['all'],output['standalone']['recipe']['all'])


if __name__=='__main__':
    main()
