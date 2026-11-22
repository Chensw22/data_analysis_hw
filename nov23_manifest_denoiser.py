"""Group-purged OOF comparison of a manifest MLP and its masked reconstruction head."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
OUT = ART / 'nov23_manifest_denoiser'
BILLS = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']


def encode(raw):
    cabin = raw.Cabin.str.split('/', expand=True)
    categorical = [raw[k] for k in ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']] + [cabin[0], cabin[2]]
    blocks = [pd.get_dummies(s.astype('string').fillna('missing'), dtype=float).to_numpy() for s in categorical]
    numeric = [(raw.Age / 80, [0,.15,.25,.4,.65]),
               (pd.to_numeric(cabin[1]) / 2000, [0,.2,.4,.6,.8])]
    for k in BILLS:
        numeric.append((np.log1p(raw[k])/10, [0,.2,.4,.6,.8]))
    for values, knots in numeric:
        v = values.fillna(0).to_numpy()
        blocks.append(np.column_stack([values.isna(), values.eq(0), v,
                                       *[np.maximum(0, v-k) for k in knots[1:]]]))
    x = np.column_stack(blocks)
    field = np.repeat(np.arange(len(blocks)), [a.shape[1] for a in blocks])
    group = raw.PassengerId.str.split('_').str[0]
    size = group.groupby(group).transform('size').to_numpy()
    # Mean of other ticket members' manifest fields; only publicly supplied attributes.
    peer = (pd.DataFrame(x).groupby(group).transform('sum').to_numpy()-x)/np.maximum(size[:,None]-1,1)
    return (torch.tensor(x,dtype=torch.float32),
            torch.tensor(np.column_stack([peer, np.log(size)]),dtype=torch.float32),
            torch.tensor(field), group.to_numpy())


class ManifestNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.peer = nn.Sequential(nn.Linear(width+1,24),nn.LayerNorm(24),nn.SiLU())
        self.encoder = nn.Sequential(nn.Linear(width+24,112), nn.LayerNorm(112),nn.SiLU(),nn.Dropout(.12),
                                     nn.Linear(112,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.12))
        self.outcome = nn.Linear(80,1)
        self.decoder = nn.Linear(80,width)

    def encode(self, x, peers):
        return self.encoder(torch.cat([x,self.peer(peers)],1))


def fit(x, peers, field, target, t, q, seed, weight):
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    masks = torch.Generator().manual_seed(seed+130)
    model = ManifestNet(x.shape[1])
    optim = torch.optim.AdamW(model.parameters(), lr=.0011, weight_decay=.014)
    snapshots = []
    for epoch in range(32):
        model.train()
        for idx in torch.randperm(len(t),generator=shuffle).split(192):
            batch = t[idx]
            a,b = x[batch],peers[batch]
            rep = model.encode(a,b)
            loss = F.binary_cross_entropy_with_logits(model.outcome(rep).ravel(),target[batch])
            if weight:
                mask = torch.rand((len(a), int(field.max())+1), generator=masks) < .28
                hidden = mask[:,field]
                corrupted = model.encode(torch.where(hidden,-1.,a),b)
                recon = model.decoder(corrupted)
                loss = loss + weight * (F.mse_loss(recon[hidden],a[hidden]) if hidden.any() else recon.sum()*0)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch in [15,23,31]:
            model.eval()
            with torch.inference_mode():
                snapshots.append(model.outcome(model.encode(x[q],peers[q])).ravel().sigmoid().numpy())
    return np.mean(snapshots,axis=0)


def metrics(p, labels, incumbent, hard, fold):
    d = p >= .5
    fix = (d==labels)&(incumbent!=labels)
    harm = (d!=labels)&(incumbent==labels)
    return dict(accuracy=float((d==labels).mean()), fixed=int(fix.sum()), harmed=int(harm.sum()),
                hard_fixed=int((fix&hard).sum()),hard_harmed=int((harm&hard).sum()),
                fold_net=[int(fix[fold==k].sum())-int(harm[fold==k].sum()) for k in range(3)])


def run():
    torch.set_num_threads(2)
    frame = pd.read_csv(ART/'nov21_case_checks/v136_final_oof.csv')
    raw = pd.concat([pd.read_csv(ROOT / 'data/raw' / fn, usecols=lambda s:s not in {'Name','Transported'})
                     for fn in ['train.csv','test.csv']],ignore_index=True)
    row = frame.row.to_numpy()
    y = frame.true.to_numpy()
    fold = frame.fold.to_numpy()
    anchor = frame.v131.to_numpy()
    hard = frame.hard.to_numpy()
    with np.load(ART/'v108_exact_oof/oof.npz') as z:
        np.testing.assert_equal(row,z['dev_ids'])
        np.testing.assert_equal(fold,z['fold_id'])
    np.testing.assert_equal(raw.PassengerId.to_numpy()[row],frame.PassengerId.to_numpy())
    x, peers, field, groups = encode(raw)
    x, peers, groups = x[row], peers[row], groups[row]
    target = torch.tensor(y,dtype=torch.float32)
    predictions, results = {}, {}
    for version,weight in [('supervised',0),('masked',1.0)]:
        p = np.zeros(len(y))
        for k in range(3):
            q = np.flatnonzero(fold==k)
            t = np.flatnonzero((fold!=k)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[q],groups[t]).size
            p[q] = fit(x,peers,field,target,t,q,2323+k,weight)
            print(version,k,len(t),round(float(((p[q]>=.5)==y[q]).mean()),6),flush=True)
        results[version] = metrics(p,y,anchor,hard,fold)
        predictions[version]=p
        print(version,results[version],flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'oof.npz',row=row,y=y,fold=fold,p=predictions['masked'],**predictions)
    hard_flips = frame.loc[hard & (anchor!=y) & ((predictions['masked']>=.5)==y),['row','PassengerId','fold','true','v131']]
    hard_flips.to_csv(OUT/'common_errors_recovered.csv',index=False)
    folder = ROOT/'reports/nov23_manifest_denoiser'
    folder.mkdir(exist_ok=True)
    (folder/'result.json').write_text(json.dumps(dict(rows=len(y),results=results,
        scope='Full passenger cohort, original OOF rows; all held-out booking groups removed during fitting. Dummies and peer summaries use unlabeled roster. Three fixed epoch snapshots per fold. No name or withheld outcome labels used.',
        oof='artifacts/nov23_manifest_denoiser/oof.npz'),indent=2)+'\n')


if __name__=='__main__':
    run()
