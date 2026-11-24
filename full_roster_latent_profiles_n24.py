"""Model a passenger as a mixture of latent categorical profiles, with or without class sharing."""
import json

import numpy as np
from scipy.special import expit,logsumexp,softmax
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_dependency_tree_n24 import encode
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets

MODES = ('shared_profiles','class_profiles','random_groups')


def component_scores(x,weights,tables):
    scores = np.broadcast_to(weights,(len(x),len(weights))).copy()
    for j,table in enumerate(tables):
        scores += table[x[:,j]]
    return scores


def fit_components(x,labels,dims,t,k,include_label,learn):
    # Initialize all groups with a different training passenger. Each EM responsibility
    # uses only the local outer training roster; probabilities for q never use its label.
    rng = np.random.default_rng(451)
    centers = rng.choice(t,k,replace=False)
    logits = np.column_stack([(x[t]==x[i]).sum(axis=1) for i in centers]).astype(float)
    r = softmax(.65*logits,axis=1) if learn else np.eye(k)[rng.integers(k,size=len(t))]
    prior = [(np.bincount(x[t,j],minlength=n)+1)/(len(t)+n) for j,n in enumerate(dims)]
    def parameters():
        count = r.sum(0)
        tables = []
        for j,n in enumerate(dims):
            obs = np.zeros((n,k))
            np.add.at(obs,x[t,j],r)
            tables.append(np.log((obs+8*prior[j][:,None])/(count+8)))
        label_p = (r.T@labels[t]+1)/(count+2)
        return np.log((count+1)/(len(t)+k)),tables,label_p
    if learn:
        for _ in range(40):
            w,tables,prob = parameters()
            logp = component_scores(x[t],w,tables)
            if include_label:
                logp += np.log(np.where(labels[t,None],prob,1-prob))
            updated = softmax(logp,axis=1)
            delta = np.abs(updated-r).mean()
            r = updated
            if delta<1e-5:
                break
    return parameters()


def mixtures(x,labels,dims,t,q,k,include_label,learn):
    w,tables,prob = fit_components(x,labels,dims,t,k,include_label,learn)
    scores = component_scores(x[q],w,tables)
    return softmax(scores,axis=1)@prob,logsumexp(scores,axis=1)


def fit(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    dims = x.max(0)+1
    shared = mixtures(x,y,dims,t,q,8,True,True)[0]
    out = [shared]
    for learn in (True,False):
        ll = []
        for c in (False,True):
            members = t[y[t]==c]
            obs = mixtures(x,y,dims,members,q,4,False,learn)[1]
            ll.append(np.log((len(members)+1)/(len(t)+2))+obs)
        out.append(expit(ll[1]-ll[0]))
    return np.column_stack(out)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,val in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],val)
        hard = z['remaining']
    x = encode(raw,g)
    result = {}
    specs = [('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),('historical',rows,of,champ,old[:,0])]
    for (tag,ix,f,base,p),(six,archive) in zip(specs,panels,strict=True):
        scores = protocol(x,y,g,ix,f,tag,fit,MODES,'latent_profiles')
        agree = ((six>=.5)==base[:,None]).all(1)
        all_agree = agree & ((archive>=.5)==base[:,None]).all(1)
        result[tag] = dict(archive_remaining=int((all_agree & (base!=y[ix])).sum()),methods={})
        for j,name in enumerate(MODES):
            for w in (1.,.25):
                pred = scores[:,j]>=.5 if w==1 else np.where((p>=.5)==base,(1-w)*p+w*scores[:,j]>=.5,base)
                r = stats(pred,base,y[ix],g[ix],f)
                fixed = (pred==y[ix]) & (base!=y[ix])
                r.update(six_fixed=int((fixed&agree).sum()),archive_fixed=int((fixed&all_agree).sum()),
                         hard_fixed=int((fixed&hard[ix]).sum()))
                if tag=='full':
                    new = ~np.isin(g[ix],og)
                    r['new_orders_net'] = int((pred[new]==y[ix][new]).sum()-(base[new]==y[ix][new]).sum())
                result[tag]['methods'][f'{name}_{w}'] = r
                print(tag,name,w,r['corrected'],r['harmed'],r['six_fixed'],r['archive_fixed'],r['hard_fixed'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/latent_profiles.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
