"""Learn symmetric same-outcome probabilities and compare passengers to known reference labels."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from full_roster_class_neighborhood_mass_n24 import run

MODES = ('point', 'joint_point', 'near_separable', 'near_relational', 'global_relational')


class Verifier(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width,56),nn.SiLU(),nn.Linear(56,24),nn.SiLU())
        self.point = nn.Linear(24,1)
        self.relation = nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Linear(32,1))

    def comparison(self,a,b):
        return self.relation(torch.cat([(a-b).abs(),a*b],dim=-1)).squeeze(-1)


def fit_network(x,y,g,t,near,paired):
    torch.manual_seed(402)
    rng = np.random.default_rng(624)
    net = Verifier(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.02)
    for epoch in range(40):
        for batch in np.array_split(rng.permutation(len(t)),int(np.ceil(len(t)/256))):
            people = t[batch]
            h = net.body(x[people])
            loss = F.binary_cross_entropy_with_logits(net.point(h).flatten(),y[people])
            if paired:
                refs = np.column_stack([rng.choice(t,(len(batch),3)),near[batch[:,None],rng.integers(0,near.shape[1],(len(batch),3))]])
                mask = torch.as_tensor(g[people,None] != g[refs])
                other = net.body(x[refs])
                target = (y[people,None]==y[refs]).float()
                compare = net.comparison(h[:,None,:].expand_as(other),other)
                loss = loss+F.binary_cross_entropy_with_logits(compare[mask],target[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def evidence(h,prob,net,ask,refs,relational):
    output = []
    for start in range(0,len(ask),128):
        ids = ask[start:start+128]
        chosen = refs[start:start+128]
        if relational:
            other = h[chosen]
            score = net.comparison(h[ids,None,:].expand_as(other),other)
        else:
            pa,pb = prob[ids,None],prob[chosen]
            agree = pa*pb+(1-pa)*(1-pb)
            score = torch.logit(agree.clamp(.0001,.9999))
        mid = chosen.shape[1]//2
        # Negative-labelled anchors enter in the first block, positive-labelled in the second.
        output.append(torch.stack([score[:,:mid].mean(1),score[:,mid:].mean(1),
                                   score[:,:mid].std(1),score[:,mid:].std(1)],dim=1).numpy())
    return np.vstack(output)


def predict(x,y,g,t,q):
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=819).split(t,groups=g[t]))
    train,cal = t[a],t[b]
    assert not np.isin(g[train],g[cal]).any() and not np.isin(g[t],g[q]).any()
    z = ((x-x[train].mean(0))/x[train].std(0).clip(.2)).clip(-7,7)
    tensor = torch.tensor(z,dtype=torch.float32)
    truth = torch.tensor(y,dtype=torch.float32)
    ask = np.r_[cal,q]
    near = train[cKDTree(z[train]).query(z[train],k=33)[1][:,1:]]
    rng = np.random.default_rng(754)
    global_refs,local_refs = [],[]
    for label in (False,True):
        members = train[y[train]==label]
        global_refs.append(np.tile(rng.choice(members,64,replace=False),(len(ask),1)))
        local_refs.append(members[cKDTree(z[members]).query(z[ask],k=32)[1]])
    close,far = np.column_stack(local_refs),np.column_stack(global_refs)
    matrices = []
    for paired in (False,True):
        net = fit_network(tensor,truth,g,train,near,paired)
        with torch.inference_mode():
            h = net.body(tensor)
            unary = net.point(h).flatten()
            binary = torch.sigmoid(unary)
            raw = unary[ask,None].numpy()
            matrices.append(raw)
            if paired:
                for reference,relational in [(close,False),(close,True),(far,True)]:
                    summary = evidence(h,binary,net,ask,reference,relational)
                    matrices.append(np.column_stack([raw,summary]))
    output = []
    for v in matrices:
        model = make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=300))
        model.fit(v[:len(cal)],y[cal])
        output.append(model.predict_proba(v[len(cal):])[:,1])
    return np.column_stack(output)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(predict,MODES,'labelled_reference_pairs')
