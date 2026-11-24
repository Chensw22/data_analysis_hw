"""Compare additive, any-event and odd-event response links on the same passenger fields."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, BILLS, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats

MODES = ('additive','any_event','odd_events')


def field_blocks(raw, group):
    cabin = raw.Cabin.str.split('/',expand=True)
    age = raw.Age
    grouped = raw.groupby(group)
    size = grouped.PassengerId.transform('size')
    def cats(frame):
        return pd.get_dummies(frame.astype('string').fillna('?'),dtype=float).to_numpy()
    fees = raw[BILLS]
    return [
        np.column_stack([cats(raw[['HomePlanet','Destination','VIP']]),age.fillna(30),age.isna()]),
        np.column_stack([cats(cabin[[0,2]]), pd.to_numeric(cabin[1]).fillna(-1), size,
                         grouped.Age.transform('mean').fillna(30), raw.PassengerId.str.split('_').str[1].astype(float)]),
        np.column_stack([raw.CryoSleep.eq(True), raw.CryoSleep.isna(), fees.gt(0), fees.isna()]),
        np.column_stack([np.log1p(fees.fillna(0)),np.log1p(grouped[BILLS].transform('mean').fillna(0))])
    ]


class Channels(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.views = nn.ModuleList(nn.Sequential(nn.Linear(w,28),nn.SiLU(),nn.Linear(28,1)) for w in widths)

    def forward(self, x, mode):
        logits = torch.cat([net(block) for net,block in zip(self.views,x)],dim=1)
        if mode == 'additive':
            return torch.sigmoid(logits.sum(1))
        trigger = torch.sigmoid(logits-1.5)
        if mode == 'any_event':
            return 1-(1-trigger).prod(1)
        return (1-(1-2*trigger).prod(1))/2


def fit(blocks,y,group,t,q):
    assert not np.isin(group[t],group[q]).any()
    x = [torch.from_numpy(((v-v[t].mean(0))/v[t].std(0).clip(.15)).clip(-7,7).astype('float32')) for v in blocks]
    truth = torch.tensor(y,dtype=torch.float32)
    scores = []
    for mode in MODES:
        torch.manual_seed(644)
        rng = np.random.default_rng(650)
        model = Channels([a.shape[1] for a in x])
        opt = torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        for epoch in range(28):
            for batch in np.array_split(rng.permutation(t),int(np.ceil(len(t)/384))):
                p = model([a[batch] for a in x], mode)
                loss = nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),truth[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.inference_mode():
            scores.append(model([a[q] for a in x],mode).numpy())
    return np.column_stack(scores)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    anchor = full[:,names.index('mitra96')]
    x = field_blocks(raw,g)
    a,b = [protocol(x,y,g,r,f,tag,fit,MODES,'response_channels') for r,f,tag in
           [(rows,of,'historical'),(np.arange(len(y)),fold,'full')]]
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    unseen = ~np.isin(g,og)
    report = {}
    for i,name in enumerate(MODES):
        for w in (1,.25):
            def decision(soft,base,challenger):
                mixed = ((1-w)*soft+w*challenger)>=.5
                return mixed if w==1 else np.where((soft>=.5)==base,mixed,base)
            pred = decision(anchor,ref,b[:,i])
            past = decision(old[:,0],champ,a[:,i])
            report[f'{name}_{w}'] = dict(full=stats(pred,ref,y,g,fold),history=stats(past,champ,oy,og,of),
                        unseen=stats(pred[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                        hard_fixed=int((hard & (pred==y)).sum()),solo_hard_fixed=int((hard & solo & (pred==y)).sum()))
            r = report[f'{name}_{w}']
            print(name,w,r['full']['corrected'],r['full']['harmed'],r['history']['net'],r['hard_fixed'],flush=True)
    report['direct_union_hard_fixed'] = int((hard & ((b>=.5)==y[:,None]).any(1)).sum())
    (ROOT/'reports/nov23_matched_learning_curve/response_channels.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run()
