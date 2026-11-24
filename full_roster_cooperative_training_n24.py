"""Train a neural challenger directly on the loss of its combination with an independent base."""
import json
from math import ceil

import numpy as np
import torch
from scipy.special import expit,logit
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_maxout_n24 import Classifier,fields


def fit(train_x,query_x,target,teacher,mode,seed):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    center,scale=train_x.mean(0),train_x.std(0).clip(.25,None)
    train=torch.from_numpy(((train_x-center)/scale).astype('float32'))
    query=torch.from_numpy(((query_x-center)/scale).astype('float32'))
    y=torch.tensor(target,dtype=torch.float32)
    p=teacher.copy()
    if mode=='shuffled_mixture':
        # Preserve the teacher's class-conditional distribution of confidence,
        # breaking only the link from its difficulty to this particular training passenger.
        for k in (False,True):
            ids=np.flatnonzero(target==k)
            p[ids]=rng.permutation(p[ids])
    p=torch.tensor(p,dtype=torch.float32).clamp(.001,.999)
    model=Classifier(train.shape[1],'maxout')
    opt=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.025)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=70,eta_min=.0004)
    snapshots=[]
    for epoch in range(70):
        for ids in np.array_split(rng.permutation(len(y)),ceil(len(y)/384)):
            z=model(train[ids])
            if mode=='standalone':
                loss=F.binary_cross_entropy_with_logits(z,y[ids])
            elif mode=='offset':
                loss=F.binary_cross_entropy_with_logits(torch.logit(p[ids])+z,y[ids])
            else:
                loss=F.binary_cross_entropy((p[ids]+torch.sigmoid(z))/2,y[ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
        schedule.step()
        if epoch in [49,59,69]:
            with torch.inference_mode():
                snapshots.append(model(query).numpy())
    return np.mean(snapshots,axis=0)


def run():
    _,y,g,rows,oldfold,newfold,champ,_,_,_,_,old,_=aligned()
    raw,labels,groups,fold=data()
    names,half,full=banks(raw,labels,fold)
    ref=latest_reference(labels,groups,fold,names,half,full)
    x=fields(raw)
    teacher=half[:,4]
    current=full[:,4]
    historical=old[:,0]
    def augment(attr,p):
        return np.column_stack([attr,logit(p.clip(.005,.995))/3])
    shared=((full>=.5)!=labels[:,None]).all(1)
    hard=shared[rows]&(champ!=y)
    unseen=~np.isin(groups,g)
    assert unseen.sum()==1044 and hard.sum()==201
    output={}
    for mode in ['standalone','mixture','shuffled_mixture','offset']:
        old_z=np.full(len(rows),np.nan)
        new_z=np.full(len(labels),np.nan)
        for f in np.unique(fold):
            # The half-panel teacher predicting all donors excluded the whole query panel,
            # including these donors and every booking receiving a prediction here.
            t=np.flatnonzero((fold!=f)&(fold%2==f%2))
            q=np.flatnonzero(fold==f)
            oq=np.flatnonzero(newfold==f)
            assert not np.isin(groups[t],groups[q]).any()
            ax=augment(x[t],teacher[t])
            bx=np.vstack([augment(x[q],current[q]),augment(x[rows[oq]],historical[oq])])
            z=np.mean([fit(ax,bx,labels[t],teacher[t],mode,seed) for seed in (405,1744)],axis=0)
            new_z[q],old_z[oq]=z[:len(q)],z[len(q):]
        assert np.isfinite(old_z).all() and np.isfinite(new_z).all()
        def predict(z,p,base,strength):
            if mode=='offset':
                prob=expit(logit(p.clip(.001,.999))+strength*z)
            else:
                prob=(1-strength)*p+strength*expit(z)
            return np.where((p>=.5)==base,prob>=.5,base)
        metrics={}
        # For the offset, unit strength matches training. For mixtures, half does.
        normal=1. if mode=='offset' else .5
        for key,w in [('as_trained_mix',normal),('half_strength',normal/2)]:
            a,b=predict(old_z,historical,champ,w),predict(new_z,current,ref,w)
            def stats(pred,anchor,y,g,split,keep):
                d=((pred==y).astype(int)-(anchor==y))[keep]
                return dict(accuracy=float((pred[keep]==y[keep]).mean()),fixed=int((d==1).sum()),
                       harmed=int((d==-1).sum()),fold_net=[int(d[split[keep]==k].sum()) for k in np.unique(split[keep])],
                       ci_pp=interval(d,g[keep]))
            metrics[key]=dict(historical=stats(a,champ,y,g,oldfold,np.ones(len(y),bool)),
                 full=stats(b,ref,labels,groups,fold,np.ones(len(labels),bool)),
                 unseen_orders=stats(b,ref,labels,groups,fold,unseen),
                 common_fixed=int((shared&(b==labels)).sum()),
                 shared_twice_fixed=int((hard&(a==y)&(b[rows]==y)).sum()))
        output[mode]=metrics
        print(mode,[(k,v['historical']['fixed'],v['historical']['harmed'],v['full']['fixed'],v['full']['harmed'])
                    for k,v in metrics.items()],flush=True)
    path=ROOT/'reports/nov23_matched_learning_curve/cooperative_training.json'
    path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
