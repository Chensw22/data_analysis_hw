"""Measure whether target-manifest density improves predictions on unseen booking groups."""
import numpy as np
import torch
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import load_roster
from broad_boosting_path_n23 import calibrated
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import design
from released_correction_audit_n23 import development


class PassengerNet(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(dimension,72),nn.LayerNorm(72),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(72,48),nn.SiLU(),nn.Linear(48,1))

    def forward(self,x):
        return self.readout(self.encode(x)).ravel()


def smooth_loss(model,x):
    h=model.encode(x)
    score=model.readout(h).ravel()
    # For a binary output, the steepest local log-odds change is along its gradient.
    direction=torch.autograd.grad(score.sum(),h,retain_graph=True)[0].detach()
    direction=.9*F.normalize(direction,dim=1)
    pp=torch.sigmoid(score.detach())
    plus=model.readout(h+direction).ravel()
    minus=model.readout(h-direction).ravel()
    cross=(F.binary_cross_entropy_with_logits(plus,pp)+F.binary_cross_entropy_with_logits(minus,pp))/2
    return cross-F.binary_cross_entropy_with_logits(score.detach(),pp)


def train_net(x,labels,train,query,support,seed):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    generator=np.random.default_rng(seed+17)
    student=PassengerNet(x.shape[1])
    optimizer=torch.optim.AdamW(student.parameters(),lr=.002,weight_decay=.03)
    snapshots=[]
    for epoch in range(24):
        for batch in np.array_split(rng.permutation(len(train)),int(np.ceil(len(train)/256))):
            logits=student(x[train[batch]])
            loss=F.binary_cross_entropy_with_logits(logits,labels[batch])
            if support is not None:
                pool=generator.choice(support,len(batch),replace=False)
                loss=loss+min(1.,(epoch+1)/5)*1.5*smooth_loss(student,x[pool])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in [17,20,23]:
            with torch.no_grad():
                snapshots.append(student(x[query]).sigmoid().numpy())
    return np.mean(snapshots,axis=0)


def main():
    torch.set_num_threads(2)
    row,y,masks=historical()
    labels,releases,_=development()
    np.testing.assert_array_equal(y,labels)
    frame,g,size,coverage=load_roster()
    test=np.arange(len(frame)-coverage['test_rows'],len(frame))
    frame['BookingSize']=size
    state=strata(frame.iloc[row])
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    modes=['supervised','train_smooth','external','target_mix']
    out=ART/'nov23_unlabelled_neural_smoothing'
    out.mkdir(exist_ok=True)
    plans,hist,refs={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        raw={k:np.zeros(len(y)) for k in modes}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g[row],g[row[q]]))
            train,query=row[t],row[q]
            assert not np.intersect1d(g[train],g[query]).size
            assert not np.intersect1d(g[test],g[row]).size
            _,phi=design(frame,train)
            phi=np.c_[phi,np.log(size)]
            x=torch.from_numpy(StandardScaler().fit(phi[train]).transform(phi).clip(-6,6).astype('float32'))
            rng=np.random.default_rng(764+int(f))
            mix=np.r_[query,rng.choice(test,len(test)-len(query),replace=False)]
            assert len(mix)==len(test)==len(np.unique(mix))
            for k,pool in zip(modes,[None,train,test,mix]):
                raw[k][q]=train_net(x,torch.from_numpy(y[t].astype('float32')),train,query,pool,524+int(f))
        votes={k:v>=.5 for k,v in raw.items()}
        for k,p in raw.items():
            x=np.c_[logit(base.clip(.002,.998)),logit(p.clip(.002,.998))]
            vote=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[row[ix]]):
                    t,q=ix[a],ix[b]
                    vote[q]=calibrated(x[t],x[q],y,t)>=.5
            votes['combined_'+k]=vote
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        stats={}
        for k,v in votes.items():
            h=v==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()))
            if seed!='v131':hist.setdefault(k,[]).append(h)
        plans[seed]=stats
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**raw)
        print(seed,stats['combined_target_mix'],flush=True)
    write(REPORT/'neural.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
