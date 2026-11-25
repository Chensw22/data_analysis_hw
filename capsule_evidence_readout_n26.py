"""Extract routing and input-block ablations from the frozen recipe; calibrate on separate bookings."""
import argparse
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from booking_capsule_routing_n26 import CACHE, OUT, ROOT, SEEDS, blocks, data, fit

PACKETS = ROOT/'artifacts/nov26_capsule_evidence'
REPORT = ROOT/'reports/nov23_matched_learning_curve/capsule_evidence_readout.json'
REPEATS = [411,821]
FIELDS = ('registration','biology','spending','party')


def reconstruct(votes,bias,steps):
    import torch
    logits = torch.zeros_like(votes[:,:,:,0])
    for k in range(steps):
        weights = logits.softmax(-1)
        sums = (weights.unsqueeze(-1)*votes).sum(1)+bias
        square = sums.square().sum(-1,keepdim=True)
        capsule = sums*(square+1e-8).sqrt()/(1+square)
        agreement = (votes*capsule.unsqueeze(1)).sum(-1)
        if k+1<steps:
            logits += agreement
    output = 9*(capsule.norm(dim=-1)[:,1]-capsule.norm(dim=-1)[:,0])
    return output,weights,agreement


def capture(x,y,t,q,seed,steps):
    import torch
    evidence = []
    def observe(module,args,output):
        if torch.is_grad_enabled() or module.__class__.__qualname__!='fit.<locals>.Net':
            return
        votes = torch.cat([layer(z).reshape(-1,2,2,8) for layer,z in zip(module.votes,args[0])],dim=1)
        full,weights,agreement = reconstruct(votes,module.bias,steps)
        torch.testing.assert_close(full,output,rtol=0,atol=1e-6)
        differences = []
        for b in range(4):
            masked = votes.clone()
            masked[:,2*b:2*b+2] = 0
            differences.append((full-reconstruct(masked,module.bias,steps)[0]).numpy())
        # Four columns each of ablation effects, routing preference, geometric agreement.
        favor = weights[:,:,1].reshape(-1,4,2).mean(2)
        alignment = (agreement[:,:,1]-agreement[:,:,0]).reshape(-1,4,2).mean(2)
        evidence.append(np.column_stack([*differences,favor.numpy(),alignment.numpy()]))
    # Observe the three existing inference calls; training and optimizer stay unchanged.
    with torch.nn.modules.module.register_module_forward_hook(observe):
        score = fit(x,y,t,q,seed,steps)
    assert len(evidence)==3
    return score,np.mean(evidence,axis=0)


def extract():
    import torch
    torch.set_num_threads(2)
    raw,y,g,f = data()
    PACKETS.mkdir(exist_ok=True)
    for k in range(8):
        t,q = np.flatnonzero(f!=k),np.flatnonzero(f==k)
        x = blocks(raw,g,t)
        for seed in SEEDS:
            info = {}
            with np.load(CACHE/f'fold{k}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['labels'],y[t])
                for mode,steps in [('fixed',1),('dynamic',3)]:
                    p,e = capture(x,y,t,q,seed,steps)
                    np.testing.assert_allclose(p,z[mode],atol=1.e-6,rtol=0)
                    info[mode] = e
            np.savez_compressed(PACKETS/f'{k}_{seed}.npz',q=q,t=t,ids=raw.PassengerId.iloc[q].to_numpy(str),
                                labels=y[t],**info)
        print('verified original scores and captured field effects',k,flush=True)


def shuffled(ids,strata,sort_key,rng):
    result = np.arange(len(ids))
    for kind in np.unique(strata[ids]):
        loc = np.flatnonzero(strata[ids]==kind)
        order = loc[np.argsort(sort_key[ids[loc]],kind='stable')]
        for cell in np.array_split(order,max(1,len(order)//12)):
            if len(cell)>1:
                shuffled_order = rng.permutation(cell)
                result[shuffled_order] = np.roll(shuffled_order,1)
    np.testing.assert_array_equal(np.sort(result),np.arange(len(ids)))
    return ids[result]


def evaluate(p,ef,y,g,f,size,repeat):
    # Both explanation models and the counterfactual control receive the same three base scores.
    strata = (size>1).astype(int)+2*(p[:,0]>=0)
    modes = ['plain','fixed','dynamic','shuffled',*FIELDS,*[f'{m}_{k}' for m in ('fixed','shuffled') for k in FIELDS]]
    decisions = {key:np.zeros(len(g),bool) for key in modes}
    probability = np.zeros((len(g),len(decisions)))
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        for j,(a,b) in enumerate(GroupKFold(4,shuffle=True,random_state=repeat).split(rows,groups=g[rows])):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            rng = np.random.default_rng(repeat+19*k+j)
            st,sq = [shuffled(i,strata,p[:,0],rng) for i in (t,q)]
            for c,mode in enumerate(decisions):
                xt,xq = p[t],p[q]
                if mode!='plain':
                    which = 0 if mode.startswith('fixed') else 1
                    et,eq = (st,sq) if mode.startswith('shuffled') else (t,q)
                    field = mode.split('_')[-1]
                    cols = np.arange(12) if field not in FIELDS else np.arange(FIELDS.index(field),12,4)
                    xt=np.column_stack([xt,ef[et,which][:,cols]])
                    xq=np.column_stack([xq,ef[eq,which][:,cols]])
                scaler = StandardScaler().fit(xt)
                model = LogisticRegression(C=.04,max_iter=400).fit(scaler.transform(xt).clip(-6,6),y[t])
                probability[q,c] = model.predict_proba(scaler.transform(xq).clip(-6,6))[:,1]
                decisions[mode][q] = probability[q,c]>=.5
    return decisions,probability


def report():
    from exhaustive_family_quorums_n26 import inputs
    raw,y,g,f = data()
    _,cur,_ = inputs()
    anchor = cur['parent']
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        prior = z['p']
    probs = np.zeros((len(y),2,2))
    evidence = np.zeros((len(y),2,12,2))
    for k in range(8):
        t,q = np.flatnonzero(f!=k),np.flatnonzero(f==k)
        for j,seed in enumerate(SEEDS):
            with np.load(CACHE/f'fold{k}_{seed}.npz') as old,np.load(PACKETS/f'{k}_{seed}.npz') as z:
                for key,ex in [('t',t),('q',q),('labels',y[t]),('ids',raw.PassengerId.iloc[q].to_numpy(str))]:
                    np.testing.assert_array_equal(z[key],ex)
                for c,mode in enumerate(('fixed','dynamic')):
                    probs[q,c,j] = old[mode]
                    evidence[q,c,:,j] = z[mode]
    assert np.isfinite(evidence).all() and np.allclose(evidence[:,0,4:8],.5)
    out = {}
    scopes = dict(full=cur['known'],common=cur['common'],new=cur['known']&cur['new'])
    def counts(reference,act,m):
        good = int((m&act&(y!=reference)).sum())
        bad = int((m&act&(y==reference)).sum())
        return [good,bad]
    for name,pp,ee in [('mean',probs.mean(-1),evidence.mean(-1)),
                        *[(str(s),probs[:,:,j],evidence[:,:,:,j]) for j,s in enumerate(SEEDS)]]:
        out[name] = {}
        p = logit(np.column_stack([prior,pp]).clip(.001,.999))
        for repeat in REPEATS:
            v,pr = evaluate(p,ee,y,g,f,size,repeat)
            extra = (v['dynamic']!=anchor)&(v['plain']==anchor)
            unique = extra&(v['fixed']==anchor)&(v['shuffled']==anchor)
            out[name][str(repeat)] = {s:dict(dynamic_vs_plain=counts(v['plain'],v['dynamic']!=v['plain'],m),
                single_field_added_over_anchor={k:counts(anchor,(v[k]!=anchor)&(v['plain']==anchor),m) for k in FIELDS},
                single_field_unique_over_controls={k:counts(anchor,(v[k]!=anchor)&(v['plain']==anchor)&(v['fixed_'+k]==anchor)&(v['shuffled_'+k]==anchor),m) for k in FIELDS},
                single_field_vs_plain={k:counts(v['plain'],v[k]!=v['plain'],m) for k in FIELDS},
                fixed_vs_plain=counts(v['plain'],v['fixed']!=v['plain'],m),
                shuffled_vs_plain=counts(v['plain'],v['shuffled']!=v['plain'],m),
                extra_versus_anchor=counts(anchor,extra,m),dynamic_only_versus_controls=counts(anchor,unique,m),
                brier=dict(zip(v,((pr[m]-y[m,None])**2).mean(0).tolist(),strict=True))) for s,m in scopes.items()}
    REPORT.write_text(json.dumps(dict(runs=out,counts='[fixed, harmed]',repeats=REPEATS,
        controls='Shared Mitra, fixed and dynamic scores; fixed and dynamic explanation matrices each have 12 columns. '
        'Shuffled explanations preserve their joint rows inside size/score-matched bins independently in training and query.'),indent=2)+'\n')
    print(out['mean'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['extract','report'])
    extract() if ap.parse_args().action=='extract' else report()
