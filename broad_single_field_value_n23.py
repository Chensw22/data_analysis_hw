"""Estimate the value of observing each original field, holding passengers and fitted model fixed."""
import numpy as np
import xgboost as xgb
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_observed_mask_shift_n23 import FIELDS,rows
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_privileged_manifest_n23 import fit
from release_extended_coverage_n23 import train_offset

REPORT=ROOT/'reports/nov23_field_robustness'
NAMES=[fields[0] if len(fields)==1 else 'Cabin' for fields in FIELDS]


def main():
    row,y,frame,g,_=load()
    blocks,miss=rows(frame)
    full=~miss.any(1)
    data=np.column_stack([*blocks,frame.BookingSize.to_numpy()])
    slices=np.split(np.arange(data.shape[1]-1),np.cumsum([b.shape[1] for b in blocks])[:-1])
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        shared_wrong=(z['votes']!=y[None,:,None]).all((0,2))
        shared_right=(z['votes']==y[None,:,None]).all((0,2))
    root=ART/'nov23_single_field_value'
    root.mkdir(exist_ok=True)
    history,bank,cohorts={},{},[]
    plans={}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        probs=np.full((len(y),1+len(slices)),np.nan)
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            model=fit(data[t],y[t])
            for j,cols in enumerate([[],*slices]):
                perturbed=data[q].copy()
                perturbed[:,cols]=np.nan
                probs[q,j]=model.predict(xgb.DMatrix(perturbed,nthread=2))
        with np.load(ART/f'nov23_observed_mask_shift/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['fold'],fold)
            np.testing.assert_allclose(z['clean'][:,0],probs[:,0],atol=1e-6,rtol=0)
        assert np.isfinite(probs).all()
        hits=(probs>=.5)==y[:,None]
        if seed!='v131':cohorts.append(hits)
        odds=logit(probs.clip(.002,.998))
        base=logit(op[:,0].clip(.002,.998))
        small=odds[:,:1]
        moves=odds[:,1:]-small
        design={'one_model':small,'one_field':np.c_[small,base[:,None]+moves]}
        votes={}
        for name,x in design.items():
            v=ref.copy()
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&(ref==(op[:,0]>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    v[q]=train_offset(base,x,y,t,q)
            votes[name]=v
        metrics={k:paired(v==y,ref==y) for k,v in votes.items()}
        field_values={}
        for j,n in enumerate(NAMES,1):
            field_values[n]={s:paired(hits[m,j],hits[m,0]) for s,m in
                [('complete',full),('complete_common_wrong',full&shared_wrong),('complete_common_right',full&shared_right)]}
        if seed!='v131':
            bank.setdefault('reference',[]).append(ref==y)
            for k,v in votes.items():bank.setdefault(k,[]).append(v==y)
        plans[seed]=dict(missing=field_values,combinations=metrics)
        np.savez_compressed(root/f'{seed}.npz',row=row,y=y,fold=fold,base=op[:,0],ref=ref,probs=probs)
        print(seed,metrics,flush=True)
    all_hits=np.stack(cohorts)
    for key,mask in [('fully_recorded',full),('complete_common_wrong',full&shared_wrong),('complete_common_right',full&shared_right)]:
        s=all_hits[:,mask].all(0)
        eligible=~all_hits[:,mask,0].any(0)
        lost=all_hits[:,mask,0].all(0)
        consistently_wrong=(~all_hits[:,mask,1:]).all(0)
        history[key]=dict(n=int(mask.sum()),tree_right_three_times=int(s[:,0].sum()),
                     tree_wrong_three_times=int(eligible.sum()),
                     newly_rescued_by_any_one_omission=int((eligible&s[:,1:].any(1)).sum()),
                     newly_lost_by_any_one_omission=int((lost&consistently_wrong.any(1)).sum()),
                     new_rescues={n:int((eligible&s[:,j]).sum()) for j,n in enumerate(NAMES,1)})
    h={k:np.stack(v) for k,v in bank.items()}
    combined={k:dict(mean_net95_pp=interval((v.astype(float)-h['reference']).mean(0),g),
                     joint_wrong=int((v.all(0)&shared_wrong).sum())) for k,v in h.items()}
    write(REPORT/'single_value.json',dict(plans=plans,coverage=history,combined=combined))


if __name__=='__main__':
    with threadpool_limits(2):main()
