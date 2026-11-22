"""Fit a latent-state passenger outcome model to complete training bookings."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from nov23_manifest_denoiser import encode, metrics

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'


class BookingMixture(nn.Module):
    def __init__(self, width, group_width):
        super().__init__()
        self.group_encoder=nn.Sequential(nn.Linear(group_width,40),nn.LayerNorm(40),nn.SiLU())
        self.gate=nn.Linear(40,4)
        self.outcome=nn.Sequential(nn.Linear(width+40,100),nn.LayerNorm(100),nn.SiLU(),
                                   nn.Dropout(.12),nn.Linear(100,64),nn.SiLU(),nn.Linear(64,4))

    def heads(self,x,context,g):
        h=self.group_encoder(context)
        return self.gate(h).log_softmax(dim=1),self.outcome(torch.cat([x,h[g]],dim=1))


def fit_predict(x,context,groups,y,t,q,objective,seed):
    torch.manual_seed(seed)
    net=BookingMixture(x.shape[1],context.shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.012)
    party_ids,local=np.unique(groups[t],return_inverse=True)
    # Batch by party so that each likelihood uses all available outcomes in its group.
    parties=[t[idx] for idx in pd.Series(np.arange(len(t))).groupby(local,sort=True).indices.values()]
    rng=np.random.default_rng(seed)
    snapshots=[]
    for epoch in range(32):
        net.train()
        order=rng.permutation(len(parties))
        for subset in np.array_split(order,int(np.ceil(len(order)/130))):
            members=[parties[i] for i in subset]
            ix=np.concatenate(members)
            g=torch.from_numpy(np.repeat(np.arange(len(subset)),[len(m) for m in members]))
            prior,outs=net.heads(x[ix],context[party_ids[subset]],g)
            li=F.logsigmoid((2*y[ix,None]-1)*outs)
            if objective=='shared':
                # log(sum_z p_z * product_i p(y_i|z,x_i)), once per booking.
                joint=torch.zeros_like(prior).index_add(0,g,li)
                loss=-torch.logsumexp(prior+joint,dim=1).sum()/len(ix)
            else:
                loss=-torch.logsumexp(prior[g]+li,dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch in [15,23,31]:
            net.eval()
            with torch.inference_mode():
                qgroups,index=np.unique(groups[q],return_inverse=True)
                g=torch.from_numpy(index)
                prior,outs=net.heads(x[q],context[qgroups],g)
                snapshots.append((prior.exp()[g]*outs.sigmoid()).sum(1).numpy())
    return np.mean(snapshots,axis=0)


def verified_refs(cache,labels):
    ret={}
    for key in ['mitra','v44']:
        report=ROOT/'reports/top5_reference_audit'/key/'summary.json'
        entry,=[r for r in json.loads(report.read_text())['audits'] if r['kind']=='group' and r['seed']==20261112]
        path=ROOT/entry['oof_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['oof_sha256']
        with np.load(path) as s:
            for name in ['dev_ids','passenger_ids','group_ids','fold_id']:
                np.testing.assert_array_equal(s[name],cache[name])
            ret[key]=s['prob']
        assert abs(np.mean((ret[key]>=.5)==labels)-entry['accuracy'])<1e-10
    return ret


def run():
    torch.set_num_threads(2)
    raw=pd.concat([pd.read_csv(ROOT/'data/raw'/f,usecols=lambda k:k not in ['Transported','Name'])
                   for f in ['train.csv','test.csv']],ignore_index=True)
    x,_,_,party=encode(raw)
    _,groups=np.unique(party,return_inverse=True)
    frame=pd.DataFrame(x.numpy())
    by=frame.groupby(groups)
    context=torch.tensor(np.column_stack([by.mean(),by.std(ddof=0),np.log1p(by.size())]),dtype=torch.float32)
    panel=pd.read_csv(ART/'nov21_case_checks/v136_final_oof.csv')
    y=panel.true.to_numpy()
    hard=panel.hard.to_numpy()
    with np.load(ART/'top5_group_phase/folds_20261112.npz') as cache:
        rows,fold=cache['dev_ids'],cache['fold_id']
        np.testing.assert_array_equal(panel.row,rows)
        np.testing.assert_array_equal(party[rows],cache['group_ids'])
        np.testing.assert_array_equal(raw.PassengerId.to_numpy()[rows],cache['passenger_ids'])
        references=verified_refs(cache,y)
    stats,primary={},{}
    xx=x[rows]
    g=groups[rows]
    target=torch.tensor(y,dtype=torch.float32)
    for scheme,outer in [('group',fold),('random_purged',panel.fold.to_numpy())]:
        out={}
        for objective in ['independent','shared']:
            p=np.zeros(len(y))
            for k in range(3):
                q=np.flatnonzero(outer==k)
                t=np.flatnonzero((outer!=k)&~np.isin(g,g[q]))
                assert not np.intersect1d(g[q],g[t]).size
                p[q]=fit_predict(xx,context,g,target,t,q,objective,23500+k)
                print(scheme,objective,k,flush=True)
            out[objective]=p
        bases=references if scheme=='group' else {'v131':panel.v131.to_numpy(float)}
        stats[scheme]={name:{key:metrics(p,y,a>=.5,hard,outer) for key,a in bases.items()} for name,p in out.items()}
        stats[scheme]['shared_vs_independent']=metrics(out['shared'],y,out['independent']>=.5,hard,outer)
        folder=ART/'nov23_booking_mixture'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(scheme+'.npz'),row=rows,fold=outer,y=y,p=out['shared'],**out)
        print(scheme,stats[scheme],flush=True)
        report=ROOT/'reports/nov23_booking_mixture'
        report.mkdir(exist_ok=True)
        (report/'result.json').write_text(json.dumps(dict(metrics=stats,rows=len(rows),
            design='4 states, labeled members jointly marginalized once per training group. Gate uses unlabeled roster mean, standard deviation and size. Independent likelihood has identical parameters and initialization. Historical hard546 is diagnostic and was originally defined on a different split.'),indent=2)+'\n')


if __name__=='__main__':
    run()
