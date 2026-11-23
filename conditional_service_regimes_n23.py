"""Fit a three-regime conditional joint model of the service basket and passenger outcome."""
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from broad_opposing_passengers_n23 import evaluate
from broad_partner_specific_fields_n23 import encode
from broad_partition_evidence_n23 import ROOT

REPORT = ROOT/'reports/nov23_conditional_service_regimes'


class Regimes(nn.Module):
    def __init__(self, personal_width, full_width):
        super().__init__()
        self.personal = nn.Sequential(nn.Linear(personal_width, 32), nn.SiLU())
        self.prior = nn.Linear(32, 3)
        self.emissions = nn.Linear(32, 3*len(BILLS)*3)
        self.outcomes = nn.Sequential(nn.Linear(full_width, 40), nn.SiLU(), nn.Linear(40, 3))

    def parts(self, x, personal, states):
        z = self.personal(personal)
        prior = self.prior(z).log_softmax(1)
        logp = self.emissions(z).reshape(len(x), 3, len(BILLS), 3).log_softmax(-1)
        ix = states.clamp(min=0)[:,None,:,None].expand(-1,3,-1,1)
        evidence = (logp.gather(3, ix).squeeze(-1)*(states[:,None,:]>=0)).sum(2)
        return prior, evidence, self.outcomes(x)


def train_models(x,personal,states,labels,t,q,sham):
    torch.manual_seed(628)
    joint = Regimes(personal.shape[1], x.shape[1])
    direct = nn.Sequential(nn.Linear(x.shape[1],40),nn.SiLU(),nn.Linear(40,1))
    optim = torch.optim.AdamW([*joint.parameters(),*direct.parameters()],lr=.006,weight_decay=.025)
    rng = np.random.default_rng(628)
    estimates = {k:[] for k in ['joint','prior','swapped','direct']}
    occupancies = []
    for epoch in range(32):
        for batch in np.array_split(rng.permutation(t), int(np.ceil(len(t)/256))):
            prior,evidence,logits = joint.parts(x[batch],personal[batch],states[batch])
            truth=labels[batch]
            sign=torch.where(truth[:,None]>0,1.,-1.)
            joint_loss=-torch.logsumexp(prior+evidence+nn.functional.logsigmoid(sign*logits),1).mean()
            direct_loss=nn.functional.binary_cross_entropy_with_logits(direct(x[batch])[:,0],truth)
            optim.zero_grad()
            (direct_loss+joint_loss).backward()
            nn.utils.clip_grad_norm_([*direct.parameters(),*joint.parameters()],4.)
            optim.step()
        if epoch in (23,27,31):
            with torch.no_grad():
                prior,evidence,logits=joint.parts(x[q],personal[q],states[q])
                _,fake,_=joint.parts(x[q],personal[q],states[sham])
                for k,log_weights in [('joint',prior+evidence),('prior',prior),('swapped',prior+fake)]:
                    estimates[k].append((log_weights.softmax(1)*logits.sigmoid()).sum(1).numpy())
                estimates['direct'].append(direct(x[q])[:,0].sigmoid().numpy())
                occupancies.append((prior+evidence).softmax(1).mean(0).numpy())
    return {k:np.mean(v,axis=0) for k,v in estimates.items()},np.mean(occupancies,axis=0)


def similar_basket(frame,groups,states,t,q):
    rng = np.random.default_rng(837)
    count=(states>0).sum(1)
    home=frame.HomePlanet.to_numpy()
    sleep=frame.CryoSleep.to_numpy()
    donor=np.empty(len(q),int)
    exact=0
    for i,person in enumerate(q):
        eligible=t[(home[t]==home[person])&(sleep[t]==sleep[person])&(count[t]==count[person])]
        if len(eligible):
            exact+=1
        else:
            eligible=t[count[t]==count[person]]
            if not len(eligible):
                eligible=t
        donor[i]=rng.choice(eligible)
    assert not np.any(groups[q]==groups[donor])
    return donor,exact


def run_fold(frame,groups,y,t,q):
    assert not np.intersect1d(groups[t],groups[q]).size
    # The last five encoded fields are service amounts in the manifest's fixed order.
    blocks=encode(frame,t)
    personal=blocks[:,:-len(BILLS)].flatten(1)
    x=blocks.flatten(1)
    raw=frame[BILLS].to_numpy(float)
    cut=np.nanmedian(np.where(raw[t]>0,raw[t],np.nan),axis=0)
    bins=np.where(np.isfinite(raw),(raw>0).astype(int)+(raw>cut).astype(int),-1)
    donor,exact=similar_basket(frame,groups,bins,t,q)
    p,regimes=train_models(x,personal,torch.from_numpy(bins),torch.from_numpy(y.astype(np.float32)),t,q,donor)
    return p,dict(train=len(t),query=len(q),matched_donors=exact,posterior_mean=regimes.tolist(),
                  swapped_same_basket=int((bins[q]==bins[donor]).all(1).sum()))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        evaluate('conditional_service_regimes', run_fold, REPORT)
