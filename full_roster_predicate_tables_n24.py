"""Learn interactions among triplets of threshold predicates and compare to additive controls."""
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_mistake_influence_n24 import design

MODES = ('main_effects','fixed_triplets','learned_triplets')
REPORT = ROOT/'reports/nov23_matched_learning_curve'


def load_cohort(raw,y,fold):
    with np.load(ROOT/'artifacts/nov24_remaining_cases/audit.npz') as z:
        for k,v in [('passenger',raw.PassengerId.to_numpy(str)),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        missed = z['missed']
        np.testing.assert_array_equal(missed, ((z['scores']>=.5)!=y[:,None]).all(1))
    assert missed.sum()==223
    return missed


def binary_fields(x,t):
    columns = []
    for v in x.T:
        u = np.unique(v[t])
        if len(u)<2:
            continue
        if len(u)<=10:
            cuts = .5*(u[:-1]+u[1:])
        else:
            # Quantiles among nonzero observations retain resolution for mostly-zero charges.
            nonzero = v[t][v[t]!=0]
            cuts = np.unique(np.r_[0.,np.quantile(nonzero,[.1,.25,.5,.75,.9])])
        for cut in cuts:
            yes = v>cut
            if yes[t].sum()>=10 and (~yes[t]).sum()>=10:
                columns.append(yes)
    indicators = np.column_stack(columns)
    _,keep = np.unique(np.packbits(indicators[t],axis=0),axis=1,return_index=True)
    return indicators[:,np.sort(keep)].astype('float32')*2-1


class Tables(nn.Module):
    def __init__(self,width,mode):
        super().__init__()
        self.mode = mode
        self.main = nn.Linear(width,1)
        nn.init.zeros_(self.main.weight)
        nn.init.zeros_(self.main.bias)
        if mode!='main_effects':
            gen = np.random.default_rng(636)
            ix = np.stack([gen.choice(width,24,replace=False).reshape(3,8) for _ in range(32)])
            self.register_buffer('ix',torch.from_numpy(ix))
            self.select = nn.Parameter(torch.randn(32,3,8)*.2,requires_grad=mode=='learned_triplets')
            self.interaction = nn.Parameter(torch.zeros(32,4))

    def forward(self,x,temp):
        main = self.main(x).flatten()
        if self.mode=='main_effects':
            return main
        if self.mode=='fixed_triplets':
            bits = x[:,self.ix[:,:,0]]
        else:
            # Each predicate slot interpolates between eight candidates without using labels at inference.
            prob = torch.softmax(self.select/temp,dim=-1)
            bits = (x[:,self.ix]*prob.unsqueeze(0)).sum(-1)
        a,b,c = bits.unbind(-1)
        terms = torch.stack([a*b,b*c,a*c,a*b*c],dim=-1)
        return main+(terms*self.interaction).sum(dim=(1,2))


def score(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    bits = torch.from_numpy(binary_fields(x,t))
    target = torch.tensor(y,dtype=torch.float32)
    scores = []
    for mode in MODES:
        torch.manual_seed(523)
        rng = np.random.default_rng(524)
        model = Tables(bits.shape[1],mode)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=.03)
        readings = []
        for epoch in range(48):
            temp = max(.35,1-epoch/50)
            for batch in np.array_split(rng.permutation(t),int(np.ceil(len(t)/384))):
                logits = model(bits[batch],temp)
                loss = F.binary_cross_entropy_with_logits(logits,target[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch in (31,39,47):
                with torch.inference_mode():
                    readings.append(torch.cat([torch.sigmoid(model(bits[part],temp)) for part in np.array_split(q,6)]).numpy())
        scores.append(np.mean(readings,axis=0))
    return np.column_stack(scores)


def run():
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    x = design(raw)
    a = protocol(x,y,g,rows,ofold,'historical',score,MODES,'predicate_tables')
    b = protocol(x,y,g,np.arange(len(y)),fold,'full',score,MODES,'predicate_tables')
    missed = load_cohort(raw,y,fold)
    shared = ((full>=.5)!=y[:,None]).all(1)
    hard = shared[rows] & (champ!=oy)
    unseen = ~np.isin(g,og)
    report = {}
    for j,mode in enumerate(MODES):
        report[mode] = {}
        for w in (1.,.25):
            p = (1-w)*gate+w*b[:,j]>=.5
            prev = (1-w)*old[:,0]+w*a[:,j]>=.5
            if w<1:
                prev = np.where((old[:,0]>=.5)==champ,prev,champ)
            improved = missed & (p==y)
            v = dict(historical=stats(prev,champ,oy,og,ofold),full=stats(p,ref,y,g,fold),
                     unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                     reached_223=int(improved.sum()),new_missed_positives=int((improved & y).sum()),
                     persistent=int((hard & (p[rows]==oy) & (prev==oy)).sum()),
                     corrected_ids=raw.loc[improved,'PassengerId'].tolist())
            report[mode][str(w)] = v
            print(mode,w,[v[k]['net'] for k in ('historical','full','unseen')],v['reached_223'],flush=True)
    (REPORT/'predicate_tables.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    torch.set_num_threads(1)
    run()
