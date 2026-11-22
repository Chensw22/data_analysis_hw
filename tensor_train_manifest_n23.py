"""Test multiplicative tensor-train interactions against same-size additive recurrent state."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from exaone_validation_data import compact_features
from grouped_five_way_coverage_n23 import verify
from manifest_tensor_outcomes import hat
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
RANK = 8
EPOCHS = 22


def design(x, size, position, train):
    def categorical(names):
        blocks = []
        for name in names:
            codes = x[name].to_numpy()
            levels = np.unique(codes[train])
            seen = codes[:, None] == levels
            blocks.extend((seen, ~seen.any(axis=1, keepdims=True)))
        return np.column_stack(blocks)
    knots = np.log1p([0, 1, 20, 100, 400, 850, 1600, 3500, 10000, 40000])
    blocks = [
        categorical(["HomePlanet", "Destination", "VIP"]),
        np.column_stack([hat(x.Age, [0, 5, 12, 18, 25, 35, 50, 80]), categorical(["CryoSleep"])]),
        np.column_stack([categorical(["Deck", "Side"]), hat(x.CabinNumber, [0,250,500,750,1000,1500,2000])]),
        np.column_stack([hat(size,[1,2,3,4,6,8]),hat(position,[1,2,3,4,6,8])]),
        hat(np.log1p(x[BILLS].sum(axis=1,min_count=1)), knots),
        *[hat(np.log1p(x[name]), knots) for name in BILLS],
    ]
    return [torch.as_tensor(block,dtype=torch.float32) for block in blocks]


class RecurrentTable(nn.Module):
    def __init__(self, dimensions, multiplicative):
        super().__init__()
        self.multiplicative = multiplicative
        self.cores = nn.ModuleList(nn.Linear(d,RANK*RANK) for d in dimensions)
        self.start = nn.Parameter(torch.randn(RANK)*.35)
        self.output = nn.Linear(RANK,1)
        self.skip = nn.Linear(sum(dimensions),1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, blocks):
        state = self.start.expand(len(blocks[0]),-1)
        for block, core in zip(blocks,self.cores):
            matrix = core(block).reshape(-1,RANK,RANK)
            if self.multiplicative:
                # Multiplication of successive input-dependent (identity + core) matrices.
                delta = torch.bmm(state.unsqueeze(1),matrix).squeeze(1)
            else:
                # Same local matrix, state, and learned scalars; combine by addition.
                # Center the transfer function at zero field signal to preserve the skip.
                expanded = state.unsqueeze(-1)
                delta = (torch.tanh(matrix + expanded)-torch.tanh(expanded)).sum(dim=1)
            state = state + delta / np.sqrt(RANK)
        return (self.output(state)+self.skip(torch.cat(blocks,dim=1))).squeeze(1)


def fit(blocks,y,t,q,seed,multiplicative):
    torch.manual_seed(seed)
    net = RecurrentTable([b.shape[1] for b in blocks],multiplicative)
    train_blocks = [b[t] for b in blocks]
    yy = torch.as_tensor(y[t],dtype=torch.float32)
    optimizer = torch.optim.AdamW(net.parameters(),lr=.0025,weight_decay=.020)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(t)).split(384):
            optimizer.zero_grad(set_to_none=True)
            logits = net([b[idx] for b in train_blocks])
            nn.functional.binary_cross_entropy_with_logits(logits,yy[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.)
            optimizer.step()
    with torch.no_grad():
        prediction = torch.cat([torch.sigmoid(net([b[batch] for b in blocks]))
                                for batch in np.array_split(q,4)]).numpy()
    return prediction, sum(p.numel() for p in net.parameters())


def metrics(p,base,y,stubborn,unresolved):
    ok = (p>=.5)==y
    reference = (base>=.5)==y
    return dict(accuracy=float(ok.mean()),corrected_false_negative=int((ok&~reference&y).sum()),
                corrected_false_positive=int((ok&~reference&~y).sum()),
                new_false_negative=int((~ok&reference&y).sum()),
                new_false_positive=int((~ok&reference&~y).sum()),
                persistent1070=dict(false_negative=int((ok&stubborn&y).sum()),false_positive=int((ok&stubborn&~y).sum())),
                unresolved466=dict(false_negative=int((ok&unresolved&y).sum()),false_positive=int((ok&unresolved&~y).sum())))


def run():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    y,old,history=read_panel()
    stubborn=~old.any(0)
    unresolved=stubborn&~history.any(axis=(0,1))
    panel=pd.read_csv(ART/'nov23_global_error_inventory.csv')
    row=panel.row.to_numpy()
    np.testing.assert_array_equal(panel.target,y)
    assert len(row)==6954 and stubborn.sum()==1070 and unresolved.sum()==466
    raw=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c not in ['Name','Transported'])
    reference_passengers=pd.concat([raw.PassengerId,pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId])
    roster_group=reference_passengers.str.split('_').str[0]
    raw=raw.iloc[row].reset_index(drop=True)
    parts=raw.PassengerId.str.split('_',expand=True)
    group=parts[0].to_numpy()
    size=parts[0].map(roster_group.value_counts()).to_numpy()
    position=parts[1].astype(int).to_numpy()
    x=compact_features(raw)
    out=ART/'nov23_tensor_train'
    out.mkdir(exist_ok=True)
    results,all_hits={},{}
    for seed in SEEDS:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as data:
            base,fold=data['base'],data['fold']
            verify(data,row,y,fold)
        models={name:np.full(len(y),np.nan) for name in ['additive_recurrent','tensor_train']}
        sizes=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            blocks=design(x,size,position,t)
            sizes_per_model=[]
            for name,p in models.items():
                p[q],n=fit(blocks,y,t,q,seed+int(k),name=='tensor_train')
                sizes_per_model.append(n)
            assert sizes_per_model[0]==sizes_per_model[1]
            sizes.append(n)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=base,**models)
        candidates={**models,**{name+'_blend':.75*base+.25*p for name,p in models.items()}}
        scores={}
        for name,p in candidates.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            scores[name]=metrics(p,base,y,stubborn,unresolved)
            all_hits.setdefault(name,[]).append((p>=.5)==y)
        results[str(seed)]={'scores':scores,'parameters_each_fold':sizes}
        print(seed,{name:round(m['accuracy'],5) for name,m in scores.items()},flush=True)
    stable={}
    for name,hits in all_hits.items():
        hit=np.all(hits,axis=0)
        stable[name]={group:dict(fn=int((hit&mask&y).sum()),fp=int((hit&mask&~y).sum()))
                      for group,mask in [('persistent1070',stubborn),('unresolved466',unresolved)]}
    write(ROOT/'reports/nov23_tensor_train/result.json',dict(seeds=SEEDS,rows=len(y),rank=RANK,
          epochs=EPOCHS,batch_size=384,results=results,stable=stable,
          populations={name:dict(fn=int((mask&y).sum()),fp=int((mask&~y).sum()))
                       for name,mask in [('persistent1070',stubborn),('unresolved466',unresolved)]},
          comparison='Ten semantic field blocks; bounded rank residual matrix products versus nonlinear additive state propagation with identical parameters, initialization, optimizer, epochs, batches, and additive skip. Blend weights fixed at 25% novel model.'))


if __name__=='__main__':
    with threadpool_limits(2):
        run()
