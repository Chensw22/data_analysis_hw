"""Learn which passengers are similar using true and false outcomes at matched initial risk."""
import json

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.special import expit, logit, softmax
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from deployment_booking_mix_n23 import passenger_vectors
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_same_model_calibration_n23 import learn


def matched_indices(risk, y, group, seed):
    rng = np.random.default_rng(seed)
    band = np.minimum((risk*8).astype(int),7)
    triplets = []
    for k in np.unique(band):
        for side in [False,True]:
            positive = np.flatnonzero((band==k)&(y==side))
            negative = np.flatnonzero((band==k)&(y!=side))
            if len(positive)<2 or not len(negative):
                continue
            anchor = np.repeat(positive,8)
            peer = rng.choice(positive,len(anchor))
            opponent = rng.choice(negative,len(anchor))
            keep = ((group[anchor] != group[peer]) & (group[anchor] != group[opponent])
                    & (group[peer] != group[opponent]))
            triplets.append(np.column_stack([anchor[keep],peer[keep],opponent[keep]]))
    samples = np.concatenate(triplets)
    assert (y[samples[:,0]] == y[samples[:,1]]).all() and (y[samples[:,1]] != y[samples[:,2]]).all()
    return samples


def embedding(a,b,labels,group,risk,mode,f):
    torch.manual_seed(946+int(f))
    gen = np.random.default_rng(847+int(f))
    y = labels.copy()
    if mode=='matched_shuffled':
        band = np.minimum((risk*8).astype(int),7)
        for k in np.unique(band):
            ids = np.flatnonzero(band==k)
            y[ids] = gen.permutation(y[ids])
    a,b = torch.tensor(a,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)
    target = torch.tensor(y,dtype=torch.float32)
    net = nn.Sequential(nn.Linear(a.shape[1],64),nn.SiLU(),nn.Linear(64,20))
    head = nn.Linear(20,1)
    opt = torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=.002,weight_decay=.01)
    for epoch in range(20):
        if mode=='point':
            for ids in np.array_split(gen.permutation(len(y)),20):
                logits = head(net(a[ids])).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits,target[ids])
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            triplets = matched_indices(risk,y,group,71000+100*int(f)+epoch)
            for rows in np.array_split(gen.permutation(len(triplets)),20):
                idx = triplets[rows]
                az,bz,cz = F.normalize(net(a[idx]),dim=2).unbind(1)
                loss = F.softplus(5*((az*cz).sum(1)-(az*bz).sum(1))).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    with torch.inference_mode():
        return F.normalize(net(a),dim=1).numpy(), F.normalize(net(b),dim=1).numpy()


def neighborhood(a,b,y,expected):
    dist = cdist(b,a,'sqeuclidean')
    ids = np.argpartition(dist,47,axis=1)[:,:48]
    d = np.take_along_axis(dist,ids,axis=1)
    w = softmax(-d/np.maximum(d.mean(1,keepdims=True),1e-8),axis=1)
    local = (1+48*(w*y[ids]).sum(1))/50
    prior = (1+48*(w*expected[ids]).sum(1))/50
    return local,logit(local)-logit(prior)


def run():
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    rawx = passenger_vectors(raw)
    # Use the same fitted experts to represent training and query passengers.
    # These experts were trained on the opposite panel from both.
    x = np.column_stack([rawx,logit(half[:,:5].clip(.02,.98))/2])
    anchor = logit(gate.clip(.001,.999))
    unanimous = ((full>=.5)==(full[:,0:1]>=.5)).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as saved:
        additional = np.ones(len(y),bool)
        additional[saved['development']] = False
    report = {}
    for mode in ['raw','point','matched','matched_shuffled']:
        local,bias,cal = np.full((3,len(y)),np.nan)
        for f in np.unique(fold):
            t = np.flatnonzero((fold!=f)&(fold%2==f%2))
            q = np.flatnonzero(fold==f)
            assert not np.isin(g[t],g[q]).any()
            if mode=='raw':
                a,b = x[t],x[q]
            else:
                a,b = embedding(x[t],x[q],y[t],g[t],half[t,4],mode,f)
            local[q],bias[q] = neighborhood(a,b,y[t],half[t,4])
            for aa,bb in GroupKFold(4,shuffle=True,random_state=462).split(q,groups=g[q]):
                donors,test = q[aa],q[bb]
                w = learn(bias[donors,None],y[donors],anchor[donors],np.zeros(1))
                cal[test] = expit(anchor[test]+w[0]*bias[test])
        assert all(np.isfinite(v).all() for v in [local,bias,cal])
        for rule,score in [('local',local),('fixed_residual',expit(anchor+bias)),('calibrated',cal)]:
            d = ((score>=.5)==y).astype(int)-(ref==y)
            report[mode+'_'+rule] = dict(accuracy=float(((score>=.5)==y).mean()),fixed=int((d==1).sum()),
                    harmed=int((d==-1).sum()),common_fixed=int((d[unanimous]==1).sum()),
                    unanimous_harmed=int((d[unanimous]==-1).sum()),additional_net=int(d[additional].sum()),
                    fold_net=[int(d[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(d,g))
        print(mode,report[mode+'_calibrated'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/matched_metric.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        run()
