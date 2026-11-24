"""Test supervised passenger prototypes in held-booking forest-leaf coordinates."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_mistake_influence_n24 import design
from full_roster_six_failure_archive_n24 import prediction_sets

MODES = ('forest','all_neighbors','random_signed','supervised_signed')
TAG = 'broad_error_transport_leaf_prototypes'


def pick_support(k, signs, target, offset, pick):
    budget = 16
    rng = np.random.default_rng(143)
    selected = np.zeros(len(signs),bool)
    score = offset.copy()
    for step in range(2*budget):
        # Equal class budgets in both arms; each prototype's effect has its class sign.
        available = (~selected) & (signs == (-1 if step%2==0 else 1))
        if pick:
            p = expit(score)
            numerator = (target-p) @ k * signs
            denominator = (p*(1-p)) @ (k*k) + 2.
            gain = np.maximum(numerator,0)**2/denominator
            chosen = np.argmax(np.where(available,gain,-np.inf))
            inc = np.clip(numerator[chosen]/denominator[chosen],0.,6.)
            score += inc*signs[chosen]*k[:,chosen]
        else:
            chosen = rng.choice(np.flatnonzero(available))
        selected[chosen] = True
    return np.flatnonzero(selected)


def weights(x,target,offset):
    def objective(w):
        score = offset+x@w
        likelihood = (np.logaddexp(0.,score)-target*score).sum()
        return likelihood + w@w, x.T@(expit(score)-target)+2*w
    solution = minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,
                        bounds=[(0.,None)]*x.shape[1],options={'ftol':1e-9,'maxiter':250})
    assert solution.success,solution.message
    return solution.x


def predict(x,y,g,t,q):
    blocks = [t[b] for _,b in GroupKFold(3,shuffle=True,random_state=331).split(t,groups=g[t])]
    outputs = []
    assert not np.isin(g[t],g[q]).any()
    for j,geometry in enumerate(blocks):
        pool,cal = blocks[(j+1)%3],blocks[(j+2)%3]
        assert not np.isin(g[geometry],g[np.r_[pool,cal]]).any() and not np.isin(g[cal],g[pool]).any()
        forest = RandomForestClassifier(n_estimators=72,min_samples_leaf=12,max_features=.65,
                                        random_state=413,n_jobs=2).fit(x[geometry],y[geometry])
        # Limit candidates before consulting calibration labels. All-neighbor controls use all donors.
        selected = np.random.default_rng(312).choice(len(pool),min(512,len(pool)),replace=False)
        query = np.r_[cal,q]
        raw = forest.predict_proba(x[query])[:,1].clip(.001,.999)
        enc = OneHotEncoder(handle_unknown='ignore',dtype=np.float32)
        donors = enc.fit_transform(forest.apply(x[pool])).T.tocsr()
        keys = enc.transform(forest.apply(x[query]))
        matrix,prevalence = [],[]
        for start in range(0,len(query),128):
            affinity = ((keys[start:start+128]@donors).toarray()/len(forest.estimators_))**3
            prevalence.append((affinity@y[pool]+2*y[pool].mean())/(affinity.sum(1)+2))
            matrix.append(affinity[:,selected])
        k = np.ascontiguousarray(np.vstack(matrix),dtype=float)
        sign = np.where(y[pool[selected]],1.,-1.)
        prior = logit(raw)
        result = [raw[len(cal):],np.concatenate(prevalence)[len(cal):]]
        for supervised in (False,True):
            ids = pick_support(k[:len(cal)],sign,y[cal],prior[:len(cal)],supervised)
            basis = k[:,ids]*sign[ids]
            w = weights(basis[:len(cal)],y[cal],prior[:len(cal)])
            result.append(expit(prior[len(cal):]+basis[len(cal):]@w))
        outputs.append(np.column_stack(result))
    return np.mean(outputs,axis=0)


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,b = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,b)
    _,sets = prediction_sets(raw,y,g,fold,rows,of)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for key,val in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[key],val)
        hard = z['remaining']
    out = {}
    x = design(raw)
    for (tag,idx,f,base,anchor),(six,archive) in zip([
            ('full',np.arange(len(y)),fold,ref,b[:,names.index('mitra96')]),
            ('historical',rows,of,champ,old[:,0])],sets,strict=True):
        truth = y[idx]
        pred = protocol(x,y,g,idx,f,tag,predict,MODES,TAG)
        missed = (base!=truth)&((six>=.5)==base[:,None]).all(1)
        all_wrong = missed & ((archive>=.5)==base[:,None]).all(1)
        if tag=='full':
            assert all_wrong.sum()==240
        results = {}
        for j,key in enumerate(MODES):
            direct = pred[:,j]
            shift = logit(direct.clip(.001,.999))-logit(pred[:,0].clip(.001,.999))
            free = (anchor>=.5)==base
            rules = dict(direct=direct>=.5, quarter=np.where(free,.75*anchor+.25*direct>=.5,base))
            if j>=2:
                rules['offset'] = np.where(free,logit(anchor.clip(.001,.999))+shift>=0,base)
            for mode,decision in rules.items():
                fixed = (base!=truth) & (decision==truth)
                result = stats(decision,base,truth,g[idx],f)
                result.update(fixed_68=int((fixed&all_wrong).sum()),fixed_six=int((fixed&missed).sum()),
                              fixed_147_in_scope=int((fixed & hard[idx]).sum()))
                if tag=='full':
                    fresh = ~np.isin(g[idx],og)
                    result['new_orders'] = stats(decision[fresh],base[fresh],truth[fresh],g[idx][fresh],f[fresh])
                results[key+'_'+mode] = result
        out[tag] = dict(missed_68=int(all_wrong.sum()),baseline_wrong=int((base!=truth).sum()),models=results)
        for key,r in results.items():
            print(tag,key,r['corrected'],r['harmed'],r['fixed_68'],r['fixed_147_in_scope'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
