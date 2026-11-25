"""Assign an unlabeled batch of passengers to outcome-specific categorical profiles."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_dependency_tree_n24 import encode
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets

TAG = 'broad_error_transport_profile_alignment'
MODES = ('independent_fixed','independent_learned','class_balanced','profile_relaxed','profile_balanced')


def categorical_profiles(x,y,source,users):
    # Cluster each outcome separately. Each observed category contributes one one-hot block.
    size = x.max(0)+1
    onehot = np.column_stack([np.eye(n)[x[:,j]] for j,n in enumerate(size)])
    widths = 12
    columns,frequency = [],[]
    for c in (False,True):
        members = source[y[source]==c]
        division = KMeans(n_clusters=widths,random_state=797,n_init=3).fit_predict(onehot[members])
        count = np.bincount(division,minlength=widths)
        assert (count>0).all()
        frequency.extend(count)
        fields = []
        for j,n in enumerate(size):
            observed = x[members,j]
            prior = np.bincount(observed,minlength=n)+1
            table = np.bincount(division*n+observed,minlength=widths*n).reshape(widths,n)
            likelihood = (table+12*prior[None,:]/prior.sum())/(count[:,None]+12)
            fields.append(np.log(likelihood[:,x[users,j]].T))
        columns.append(np.stack(fields,axis=2))
    return np.concatenate(columns,axis=1),np.asarray(frequency)/len(source)


def learn_metric(terms,prior,target):
    k = len(prior)//2
    def optimize(w):
        logits = terms@w + np.log(prior)
        conditional = np.stack([logsumexp(logits[:,:k],axis=1),logsumexp(logits[:,k:],axis=1)],axis=1)
        nll = logsumexp(conditional,axis=1)-conditional[np.arange(len(target)),target.astype(int)]
        prob = softmax(logits,axis=1)
        within = np.column_stack([softmax(logits[:,:k],axis=1)*(~target)[:,None],
                                  softmax(logits[:,k:],axis=1)*target[:,None]])
        dev = w-1.
        return nll.sum()+12*dev@dev, np.einsum('nk,nkf->f',prob-within,terms)+24*dev
    solution = minimize(optimize,np.ones(terms.shape[2]),jac=True,method='L-BFGS-B',
                        bounds=[(0.,3.)]*terms.shape[2],options={'ftol':1e-9})
    assert solution.success,solution.message
    return solution.x


def transport(logits,prior,penalty):
    # Entropic transport with equal passenger weights. The optional KL penalty
    # allows the inferred target profile masses to differ from source masses.
    def objective(offset):
        s = logits + offset
        data = logsumexp(s,axis=1).mean()
        gradient = softmax(s,axis=1).mean(0)
        if penalty is None:
            return data-prior@offset,gradient-prior
        factor = np.exp(-offset/penalty)
        return data+penalty*(prior@(factor-1)),gradient-prior*factor
    solution = minimize(objective,np.zeros(len(prior)),jac=True,method='L-BFGS-B',
                        options={'ftol':1e-14,'gtol':1e-8,'maxiter':500})
    assert solution.success,solution.message
    match = softmax(logits+solution.x,axis=1)
    expected = prior if penalty is None else prior*np.exp(-solution.x/penalty)
    np.testing.assert_allclose(match.mean(0),expected,atol=2e-5)
    return match


def check_alignment():
    prior = np.array([.25,.25,.5])
    logits = np.array([[-2,1,3],[2,2,-1],[3,-4,0],[0,1,2.]])
    fitted = transport(logits,prior,None)
    # Entropic reweighting cannot change conditional cross-odds for a pair of rows and columns.
    np.testing.assert_allclose(np.diff(np.diff(np.log(fitted),axis=1),axis=0),
                               np.diff(np.diff(logits,axis=1),axis=0),atol=1e-9)
    np.testing.assert_allclose(transport(logits[::-1],prior,None)[::-1],fitted,atol=1e-8)
    expected = np.tile(np.log(prior),(4,1))
    np.testing.assert_allclose(transport(expected,prior,2.),np.exp(expected),atol=1e-9)


def predict(x,y,g,t,q):
    assert not np.isin(g[t],g[q]).any()
    rows = []
    for a,b in GroupKFold(2,shuffle=True,random_state=196).split(t,groups=g[t]):
        source,cal = t[a],t[b]
        assert not np.isin(g[source],g[cal]).any()
        terms,prior = categorical_profiles(x,y,source,np.r_[cal,q])
        w = learn_metric(terms[:len(cal)],prior,y[cal])
        independent = terms[len(cal):].sum(2) + np.log(prior)
        learned = terms[len(cal):]@w + np.log(prior)
        mid = len(prior)//2
        class_logits = np.column_stack([logsumexp(learned[:,:mid],axis=1),logsumexp(learned[:,mid:],axis=1)])
        class_mass = np.array([prior[:mid].sum(),prior[mid:].sum()])
        rows.append(np.column_stack([
            softmax(independent,axis=1)[:,mid:].sum(1),softmax(learned,axis=1)[:,mid:].sum(1),
            transport(class_logits,class_mass,None)[:,1],
            transport(learned,prior,2.)[:,mid:].sum(1),transport(learned,prior,None)[:,mid:].sum(1)]))
    return np.mean(rows,axis=0)


def run():
    check_alignment()
    raw,y,g,f = data()
    names,h,b = banks(raw,y,f)
    ref = latest_reference(y,g,f,names,h,b)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    _,panels = prediction_sets(raw,y,g,f,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    x = encode(raw,g)
    out = {}
    for (tag,idx,fold,base,anchor),(six,archive) in zip([
            ('full',np.arange(len(y)),f,ref,b[:,names.index('mitra96')]),
            ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        truth = y[idx]
        v = protocol(x,y,g,idx,fold,tag,predict,MODES,TAG)
        agreement = ((np.c_[six,archive]>=.5)==base[:,None]).all(1)
        if tag=='full':
            assert (agreement & (base!=truth)).sum()==240
        out[tag] = {}
        for j,key in enumerate(MODES):
            for name,weight in [('direct',1.),('quarter',.25)]:
                def rule(prob):
                    if weight==1:
                        return prob>=.5
                    return np.where((anchor>=.5)==base,(1-weight)*anchor+weight*prob>=.5,base)
                pred = rule(v[:,j])
                r = stats(pred,base,truth,g[idx],fold)
                r['versus_independent_learned'] = stats(pred,rule(v[:,1]),truth,g[idx],fold)
                for k,scope in [('unanimous',agreement),('disagreement',~agreement),('hard',hard[idx])]:
                    r[k] = dict(corrected=int((scope & (base!=truth) & (pred==truth)).sum()),
                                harmed=int((scope & (base==truth) & (pred!=truth)).sum()))
                if tag=='full':
                    new = ~np.isin(g[idx],og)
                    r['new_orders'] = stats(pred[new],base[new],truth[new],g[idx][new],fold[new])
                out[tag][key+'_'+name] = r
                print(tag,key,name,r['corrected'],r['harmed'],r['unanimous']['corrected'],r['hard']['corrected'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    path.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
