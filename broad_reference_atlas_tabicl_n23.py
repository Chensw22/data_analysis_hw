"""Give a frozen classifier outcome-specific comparison profiles from independent bookings."""
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_class_hull_evidence_n23 import evidence
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import REPORT,cohorts,coordinates
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_pretrained_spending_views_n23 import predict
from tabicl_routed_context_n23 import DIGEST,WEIGHT


def profile_table(raw_table,frame,reference,rows,y):
    coords,_=coordinates(frame,reference)
    distances=evidence(coords,reference,rows,y)
    # Keep the original numeric values visible. New columns describe both outcome classes.
    values={}
    for j,name in [(0,'radius'),(2,'linear_mean'),(3,'linear_hull'),(5,'kernel_mean'),(6,'kernel_hull')]:
        a,b=distances[:,:,j].T
        values[name+'_difference']=a-b
        values[name+'_overlap']=np.minimum(a,b)
    return pd.concat([raw_table.iloc[rows].reset_index(drop=True),pd.DataFrame(values)],axis=1)


def fit_split(frame,g,y,t,q,f):
    assert not np.intersect1d(g[t],g[q]).size
    raw=frame.drop(columns='BookingSize')
    views={k:[] for k in ('plain','real','shuffled')}
    for a,b in GroupKFold(2,shuffle=True,random_state=183).split(t,groups=g[t]):
        reference,context=t[a],t[b]
        assert not np.intersect1d(g[reference],g[context]).size
        rows=np.r_[context,q]
        source=np.arange(len(context))
        query=np.arange(len(context),len(rows))
        fake=y.copy()
        rng=np.random.default_rng(68+int(f))
        fake[reference]=rng.permutation(y[reference])
        for name,labels in [('real',y),('shuffled',fake),('plain',None)]:
            table=(raw.iloc[rows].reset_index(drop=True) if labels is None
                   else profile_table(raw,frame,reference,rows,labels))
            # All context labels used by TabICL are external to the reference library.
            views[name].append(predict(table,y[context],source,query,f))
    return {k:np.mean(v,axis=0) for k,v in views.items()}


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    output=ART/'nov23_reference_atlas_tabicl'
    output.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            file=output/f'{seed}_{f}.npz'
            if file.is_file():
                with np.load(file) as z:
                    for k,v in [('t',row[t]),('q',row[q]),('yt',y[t])]:np.testing.assert_array_equal(z[k],v)
                    preds={k:z[k] for k in ['plain','real','shuffled']}
            else:
                preds=fit_split(frame,g,y,t,q,f)
                np.savez_compressed(file,t=row[t],q=row[q],yt=y[t],**preds)
            for k,v in preds.items():scores.setdefault(k,np.full(len(y),np.nan))[q]=v
        hits={'reference':ref==y}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            hits[k]=(p>=.5)==y
            hits['quarter_'+k]=np.where((op[:,0]>=.5)==ref,.75*op[:,0]+.25*p>=.5,ref)==y
        changes={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(vs_reference=changes,vs_plain={k:paired(v,hits['plain']) for k,v in hits.items()})
        if seed!='v131':
            for k,v in hits.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(output/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],**scores)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in changes.items()},flush=True)
    ref=np.stack(hist['reference'])
    stable={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                   hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                   gain95_pp=interval((np.stack(v).astype(float)-ref).mean(0),g)) for k,v in hist.items()}
    write(REPORT/'foundation.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):main()
