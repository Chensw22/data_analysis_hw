"""Compare additive, pairwise, and third-order field interactions on the full passenger roster."""
import json

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from exaone_validation_data import CATEGORICAL, compact_features
from full_roster_cohort_prevalence_n23 import fraction
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities


def encode(frame, train):
    indices, weights = [], []
    offset = 0
    for name in frame:
        column = frame[name]
        if name in CATEGORICAL:
            levels = {key:i+1 for i,key in enumerate(sorted(set(column.iloc[train])))}
            tok = column.map(levels).fillna(0).to_numpy(dtype=int)
            left, right = tok, tok
            frac = np.zeros(len(tok))
            width = len(levels)+1
        else:
            values = column.to_numpy(dtype=float)
            positive = values[train][values[train] > 0]
            knots = np.unique(np.quantile(positive,np.linspace(0,1,14))) if len(positive) else np.array([1.])
            v = np.clip(np.nan_to_num(values,nan=0.),knots[0],knots[-1])
            hi = np.minimum(np.searchsorted(knots,v),len(knots)-1)
            lo = np.maximum(hi-1,0)
            step = knots[hi]-knots[lo]
            frac = np.divide(v-knots[lo],step,out=np.zeros_like(v),where=step>0)
            left, right = lo+2, hi+2
            for special,mask in [(0,np.isnan(values)),(1,values==0)]:
                left[mask] = right[mask] = special
                frac[mask] = 0.
            width = len(knots)+2
        indices.append(np.column_stack([left+offset,right+offset]))
        weights.append(np.column_stack([1-frac,frac]))
        offset += width
    return (torch.tensor(np.stack(indices,axis=1),dtype=torch.long),
            torch.tensor(np.stack(weights,axis=1),dtype=torch.float32),offset)


class InteractionModel(nn.Module):
    def __init__(self, width, order):
        super().__init__()
        self.order = order
        self.linear = nn.Embedding(width,1)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(()))
        self.tables = nn.ModuleList(nn.Embedding(width,8) for _ in range(order-1))
        for table in self.tables:
            nn.init.normal_(table.weight,std=.12)

    def forward(self, ids, weights):
        w = weights.unsqueeze(-1)
        out = (self.linear(ids)*w).sum(dim=(1,2,3)) + self.bias
        for degree, table in enumerate(self.tables, start=2):
            factors = (table(ids)*w).sum(dim=2)
            total = factors.sum(dim=1)
            square_sum = factors.square().sum(dim=1)
            if degree == 2:
                interaction = (total.square()-square_sum)/2
            else:
                interaction = (total.pow(3)-3*total*square_sum+2*factors.pow(3).sum(dim=1))/6
            out = out + interaction.sum(dim=1)
        return out


def train_order(ids, weights, width, y, t, q, order, seed):
    torch.manual_seed(seed)
    model = InteractionModel(width,order)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.008,weight_decay=.03)
    target = torch.tensor(y.astype('float32'))
    saved = []
    for epoch in range(32):
        model.train()
        order_ids = torch.as_tensor(t)[torch.randperm(len(t))]
        for batch in order_ids.split(256):
            optimizer.zero_grad()
            logits = model(ids[batch],weights[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(logits,target[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch in (15,23,31):
            model.eval()
            with torch.no_grad():
                saved.append(torch.sigmoid(model(ids[q],weights[q])).numpy())
    return np.mean(saved,axis=0)


def run():
    torch.set_num_threads(2)
    raw,y,group,fold=data()
    names,half,full=banks(raw,y,fold)
    reference=latest_reference(y,group,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,y,group,fold)
    np.testing.assert_array_equal(gate>=.5,reference)
    frame=compact_features(raw)
    _,ix,size=np.unique(group,return_inverse=True,return_counts=True)
    frame['BookingSize']=size[ix]
    frame['Position']=raw.PassengerId.str.split('_').str[1].astype(int)
    probs={order:np.full(len(y),np.nan) for order in (1,2,3)}
    for f in np.unique(fold):
        t,q=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        assert not np.isin(group[t],group[q]).any()
        ids,weights,width=encode(frame,t)
        for order in probs:
            probs[order][q]=train_order(ids,weights,width,y,t,q,order,426+int(f))
        print('finished fold',f,flush=True)
    path=ROOT/'artifacts/nov24_field_interactions'
    path.mkdir(exist_ok=True)
    np.savez_compressed(path/'oof.npz',passenger=raw.PassengerId.to_numpy(str),fold=fold,y=y,
                        additive=probs[1],pair=probs[2],triple=probs[3])
    middle=(gate>.3)&(gate<.7)
    shared=((full>=.5)!=y[:,None]).all(1)
    with np.load(ROOT/'artifacts/run_002/split_indices.npz') as z:
        dev=np.sort(z['development'])
    result={}
    for order,p in probs.items():
        assert np.isfinite(p).all()
        adaptive=gate.copy()
        for f in np.unique(fold):
            outer=np.flatnonzero(fold==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=462).split(outer,groups=group[outer]):
                t,q=outer[a],outer[b]
                assert not np.isin(group[t],group[q]).any()
                w=fraction(gate[t],p[t],y[t])
                adaptive[q]=(1-w)*gate[q]+w*p[q]
        for method,score in [('direct',p),('quarter_blend',.75*gate+.25*p),('learned_blend',adaptive)]:
            delta=((score>=.5)==y).astype(int)-(reference==y)
            result[f'order{order}_{method}']=dict(accuracy=float(((score>=.5)==y).mean()),fixed=int((delta==1).sum()),
                harmed=int((delta==-1).sum()), middle_fixed=int((delta[middle]==1).sum()),
                middle_harmed=int((delta[middle]==-1).sum()), common_fixed=int((delta[shared]==1).sum()),
                additional_net=int(delta.sum()-delta[dev].sum()),
                fold_net=[int(delta[fold==f].sum()) for f in np.unique(fold)],ci_pp=interval(delta,group))
    (ROOT/'reports/nov23_matched_learning_curve/factor_interactions.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,v in result.items(): print(k,round(v['accuracy']*100,3),v['fixed'],v['harmed'],v['common_fixed'])


if __name__=='__main__':
    run()
