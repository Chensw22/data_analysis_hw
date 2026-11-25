"""Learn which of two similarly confident decisions failed, then restore absolute error probabilities."""
import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from torch import nn

from booking_mix_confidence_transfer_n26 import assignments
from full_roster_booking_validation_n23 import ROOT
from tabicl_consensus_review_n26 import matrices,sources


def pairs(v,t,wrong,seed):
    matches=[];placebo=[]
    rng=np.random.default_rng(seed)
    group=2*v['anchor'][t]+v['common'][t]
    # Index arithmetic is local to t. Cases and controls belong to the same calibration source.
    score=v['basic'][t][:,[2,3,4,7]]
    for c in np.unique(group):
        case=np.flatnonzero((group==c)&wrong)
        control=np.flatnonzero((group==c)&~wrong)
        if not len(case) or not len(control):
            continue
        cost=cdist(score[case],score[control],metric='cityblock')
        a,b=linear_sum_assignment(cost)
        matches.extend(zip(case[a],control[b]))
        placebo.extend(zip(case[a],rng.permutation(control)[:len(a)]))
    return np.array(matches),np.array(placebo)


def train(x,z,off,offq,wrong,pair,seed):
    t=torch.tensor(x,dtype=torch.float32);q=torch.tensor(z,dtype=torch.float32)
    p=torch.tensor(off,dtype=torch.float32);pq=torch.tensor(offq,dtype=torch.float32)
    target=torch.tensor(wrong,dtype=torch.float32)
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(x.shape[1],40),nn.LayerNorm(40),nn.SiLU(),nn.Dropout(.12),nn.Linear(40,1))
    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.04)
    rng=np.random.default_rng(seed)
    estimates=[]
    for epoch in range(20):
        net.train()
        for batch in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/192))):
            score=p[batch]+net(t[batch]).flatten()
            loss=nn.functional.binary_cross_entropy_with_logits(score,target[batch])
            if pair is not None:
                left,right=pair[rng.choice(len(pair),min(len(pair),len(batch)),replace=False)].T
                gap=p[left]-p[right]+(net(t[left])-net(t[right])).flatten()
                loss=loss+.35*nn.functional.softplus(-gap).mean()
            opt.zero_grad();loss.backward();opt.step()
        if epoch in (9,14,19):
            net.eval()
            with torch.no_grad():
                estimates.append(torch.sigmoid(pq+net(q).flatten()).numpy())
    return np.mean(estimates,axis=0)


def run():
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    probs={};tested=np.zeros(len(y),bool)
    # Register nominal levels observed in calibration; these dummies do not use the deployment population.
    for f,k,t,q in assignments(old,cur,g):
        a,b=matrices(raw,old,cur,t,q)
        x=a[1].assign(all_agree=old['common'][t].astype(int))
        z=b[1].assign(all_agree=cur['common'][q].astype(int))
        amounts=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
        for frame in [x,z]:
            frame[amounts]=np.log1p(frame[amounts])
        x=pd.get_dummies(x,dtype=float)
        z=pd.get_dummies(z,dtype=float).reindex(columns=x.columns,fill_value=0)
        x,z=x.fillna(-1).to_numpy(float),z.fillna(-1).to_numpy(float)
        center=x.mean(0);scale=x.std(0).clip(.25)
        x,z=((s-center)/scale for s in (x,z))
        actual,control=pairs(old,t,old['wrong'][t],2729)
        for name,p in [('direct',None),('matched',actual),('random',control)]:
            estimates=[train(x.clip(-6,6),z.clip(-6,6),old['off'][t],cur['off'][q],old['wrong'][t],p,seed) for seed in (420,617)]
            probs.setdefault(name,np.zeros(len(y)))[q]=np.mean(estimates,axis=0)
        tested[q]=True
    out={}
    wrong=cur['wrong']
    fresh=~np.isin(g,g[old['ids']])
    scopes={'all':tested,'unanimous':tested&cur['common'],'fresh':tested&fresh,'fresh_unanimous':tested&fresh&cur['common']}
    for n,p in probs.items():
        table={}
        for s,scope in scopes.items():
            ix=np.flatnonzero(scope); high=ix[np.argsort(-p[ix],kind='stable')[:int(np.ceil(len(ix)*.1))]]
            proposed=scope&(p>.5)
            table[s]=dict(rows=len(ix),errors=int(wrong[ix].sum()),fixed=int((proposed&wrong).sum()),
                          harmed=int((proposed&~wrong).sum()),auc=float(roc_auc_score(wrong[ix],p[ix])),
                          top_tenth=len(high),top_tenth_errors=int(wrong[high].sum()))
        out[n]=table
    print([(n,d['all']['fixed'],d['all']['harmed'],d['unanimous']) for n,d in out.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/paired_unanimous_error_rank.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    run()
