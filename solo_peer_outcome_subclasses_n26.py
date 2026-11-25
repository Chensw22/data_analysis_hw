"""Share four latent outcome subclasses between solo and group passengers using peer labels only during training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import log_loss

from exaone_validation_data import compact_features, CATEGORICAL
from full_roster_booking_validation_n23 import ROOT, data

OUT = ROOT/'artifacts/nov26_solo_peer_subclasses'
BILLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def personal(frame):
    x = compact_features(frame)
    for fee in BILLS:
        x[fee+'_positive'] = x[fee]>0
        x[fee+'_missing'] = x[fee].isna()
        x[fee] = np.log1p(x[fee])/6
    x['Age'] = x.Age/30
    x['CabinNumber'] = x.CabinNumber/800
    return pd.get_dummies(x, columns=CATEGORICAL, dtype=float).fillna(-1).to_numpy(np.float32)


def peer_fraction(labels, groups):
    table = pd.Series(labels.astype(float)).groupby(groups)
    n = table.transform('size').to_numpy()-1
    total = table.transform('sum').to_numpy()-labels
    return total/n.clip(1), n


def reassign(peer,n,y,groups,seed):
    # Retain the relationship with the focal outcome and observed party size.
    # Within these strata transfer the complete fractional target between bookings.
    shuffled = peer.copy()
    rng = np.random.default_rng(seed)
    for k in np.unique(2*np.minimum(n,3)+y):
        pos = np.flatnonzero((2*np.minimum(n,3)+y==k)&(n>0))
        if not len(pos):
            continue
        levels, codes, counts = np.unique(groups[pos],return_inverse=True,return_counts=True)
        order = pos[np.argsort(rng.permutation(len(levels))[codes],kind='stable')]
        if len(pos)<2*counts.max():
            raise ValueError('Cannot derange this auxiliary-label stratum')
        donor = np.roll(order, int(counts.max()))
        assert (groups[donor]!=groups[order]).all()
        shuffled[order] = peer[donor]
    return shuffled


def predict(x,y,groups,t,q,k):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)
    peer,num = peer_fraction(y[t],groups[t])
    sham = reassign(peer,num,y[t],groups[t],31416+k)
    features = torch.as_tensor(x)
    truth = torch.as_tensor(y.astype(int))
    outputs = {}
    for mode,aux in [('plain',peer),('real',peer),('shuffled',sham)]:
        mean = []
        for seed in [1729,2213]:
            torch.manual_seed(seed+k)
            rng = np.random.default_rng(seed+k)
            net = nn.Sequential(nn.Linear(x.shape[1],96),nn.LayerNorm(96),nn.SiLU(),
                                nn.Linear(96,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,4))
            opt = torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.012)
            targets = torch.as_tensor(aux,dtype=torch.float32)
            active = torch.as_tensor(num>0)
            snapshots = []
            for epoch in range(25):
                for ix in np.array_split(rng.permutation(len(t)),int(np.ceil(len(t)/256))):
                    label = truth[t[ix]]
                    energy = net(features[t[ix]]).reshape(-1,2,2)
                    loss = F.cross_entropy(torch.logsumexp(energy,dim=2),label)
                    if mode!='plain':
                        # Factor P(Y, peerY | personal X) into the marginal focal
                        # label and the conditional peer probability given that label.
                        sub = energy[torch.arange(len(ix)),label]
                        cond = sub[:,1]-sub[:,0]
                        extra = F.binary_cross_entropy_with_logits(cond,targets[ix],reduction='none')
                        loss = loss + (extra*active[ix]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if epoch in (14,19,24):
                    with torch.no_grad():
                        joint = net(features[q]).softmax(dim=1).reshape(-1,2,2)
                        snapshots.append(joint.numpy().copy())
            mean.extend(snapshots)
        outputs[mode] = np.mean(mean,axis=0)
    return outputs


def train():
    raw,y,g,f = data()
    x = personal(raw)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf = z['rows'],z['fold']
    OUT.mkdir(exist_ok=True)
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            q = ids[fold==k]
            t = ids[(fold!=k)&~np.isin(g[ids],g[q])]
            path = OUT/f'{tag}_{k}.npz'
            if not path.exists():
                scores = predict(x,y,g,t,q,int(k))
                with path.with_suffix('.tmp').open('wb') as buf:
                    np.savez_compressed(buf,t=t,q=q,train_y=y[t],**scores)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,expected in [('t',t),('q',q),('train_y',y[t])]:
                    np.testing.assert_array_equal(z[key],expected)
                for key in ['plain','real','shuffled']:
                    np.testing.assert_allclose(z[key].sum(axis=(1,2)),1.,atol=1e-6)
            print(tag,k,flush=True)


def score():
    from disjoint_tree_fields_n25 import ART,DEST,stats
    from outstanding_error_families_n26 import observed
    from family_identity_transfer_n26 import panels_and_reference,fit
    from sklearn.model_selection import GroupKFold
    raw,y,g,f = data()
    _,_,_,rows,names,p,h,native,old_native,r=panels_and_reference()
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],r[0])
        r[0]=z['prior']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        hf,par,old=z['fold'],z['v131'],z['nine']
    awake=raw.CryoSleep.eq(False).to_numpy()
    curr=observed(np.arange(len(y)),f,p,names,native,*r,awake,'full')
    hist=observed(rows,hf,h,names,old_native,par,old,old,awake,'historical')
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    result={}
    for tag,ids,fold,base,view in [('historical',rows,hf,old_native,hist),('full',np.arange(len(y)),f,native,curr)]:
        flags={}; diagnostics={}
        fresh=~np.isin(g[ids],g[rows])
        for k in np.unique(fold):
            loc=np.flatnonzero(fold==k); q=ids[loc]
            if tag=='historical':
                fresh[loc]=~np.isin(g[q],g[ids[fold!=k]])
            peer,n=peer_fraction(y[q],g[q])
            with np.load(OUT/f'{tag}_{k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                margins={}
                for mode in ['plain','real','shuffled']:
                    joint=z[mode]
                    prob=joint[:,1,:].sum(1)
                    margins[mode]=logit(prob.clip(.002,.998))
                    anchor=view['anchor'][loc]
                    flag=(prob>=.5)!=anchor
                    flags.setdefault(mode,np.zeros(len(ids),bool))[loc]=flag
                    if mode!='plain':
                        changed=(prob>=.5)!=(margins['plain']>=0)
                        flags.setdefault(mode+'_induced',np.zeros(len(ids),bool))[loc]=flag&changed
                    conditional=joint[np.arange(len(q)),y[q].astype(int)]
                    predicted=(conditional[:,1]/conditional.sum(1)).clip(.0001,.9999)
                    diagnostic=-peer*np.log(predicted)-(1-peer)*np.log1p(-predicted)
                    diagnostics.setdefault(mode,[]).extend(diagnostic[n>0])
            off=logit(base[loc].clip(.002,.998))
            design=np.column_stack([np.ones(len(q)),off,margins['plain']-off])
            select=np.flatnonzero(solo[q])
            for t,u in GroupKFold(4,shuffle=True,random_state=2886).split(select,groups=g[q[select]]):
                tr,te=select[t],select[u]
                for mode in margins:
                    mat=design if mode=='plain' else np.column_stack([design,margins[mode]-margins['plain']])
                    beta=fit(mat[tr],off[tr],y[q[tr]])
                    new=off[te]+mat[te]@beta>=0
                    anchor=view['anchor'][loc[te]]
                    flags.setdefault(mode+'_solo_fit',np.zeros(len(ids),bool))[loc[te]]=(new!=anchor)&(anchor==(off[te]>=0))
        known=view['known']
        hard=view['eligible']&(view['votes']==view['anchor'][:,None]).all(1)
        wrong=view['anchor']!=y[ids]
        counts={}
        for mode,edits in flags.items():
            delta=edits*np.where(wrong,1,-1)
            counts[mode]={cohort:stats(delta,subset,fold,g[ids]) for cohort,subset in [
                ('solo',known&solo[ids]),('solo_common',hard&solo[ids]),('new_solo',known&solo[ids]&fresh),
                ('solo_common_new',hard&solo[ids]&fresh),('all',known)]}
        result[tag]=dict(conditional_peer_log_loss={m:float(np.mean(v)) for m,v in diagnostics.items()},results=counts)
        print(tag,[(m,c['solo']['fixed'],c['solo']['harmed'],c['solo_common']['fixed'],c['solo_common']['harmed']) for m,c in counts.items()],flush=True)
    DEST.with_name('solo_peer_outcome_subclasses.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','score'])
    {'train':train,'score':score}[p.parse_args().action]()
