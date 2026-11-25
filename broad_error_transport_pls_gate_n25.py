"""Select four-component local PLS changes using other bookings from the same outer fold."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_error_transport_local_pls_n24 import TAG
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_solo_coverage_n24 import cached


def direction(p, proposed):
    """Probability assigned to the label proposed by local PLS."""
    return np.where(proposed[:,None],p,1-p)


def feature_sets(base,anchor,pls,six,archive):
    change = (pls[:,-1]>=.5)!=base
    # The proposed direction is the actual PLS classification, including on protected changes.
    proposed = pls[:,-1]>=.5
    offset = logit(direction(anchor[:,None],proposed)[:,0].clip(.002,.998))
    support = direction(pls,proposed)
    enhanced = direction(np.column_stack([six,archive]),proposed)
    votes = (np.column_stack([six,archive])>=.5) == proposed[:,None]
    core = np.column_stack([proposed,offset,logit(support[:,-1].clip(.002,.998)),
                            (anchor>=.5)!=base])
    controls = np.column_stack([core,logit(support[:,:-1].clip(.002,.998))])
    all_opinions = np.column_stack([controls,votes[:,:6].mean(1),votes[:,6:].mean(1),
                                   enhanced[:,:6].mean(1),enhanced[:,6:].mean(1)])
    return change,offset,dict(core=core,controls=controls,panel=all_opinions)


def fit(x,offset,target):
    # Retain the raw anchor unless same-fold calibration orders provide evidence to shift it.
    def objective(w):
        z = offset + x@w[1:] + w[0]
        diff = expit(z)-target
        penalty = 8.*w[1:]@w[1:] + 2.*w[0]**2
        grad = np.r_[diff.sum()+4*w[0],x.T@diff+16*w[1:]]
        return (np.logaddexp(0,z)-target*z).sum()+penalty,grad
    fitted = minimize(objective,np.zeros(1+x.shape[1]),jac=True,method='L-BFGS-B')
    assert fitted.success,fitted.message
    return fitted.x


def route(base,anchor,pls,six,archive,y,g,f):
    dissent,offset,features = feature_sets(base,anchor,pls,six,archive)
    votes = {'all':dissent.copy()}
    support = ((pls[:,:-1]>=.5)!=(base[:,None])).sum(1)
    votes['three_controls'] = dissent & (support>=3)
    votes.update({k:np.zeros(len(y),bool) for k in features})
    seen = np.zeros(len(y),int)
    for outer in np.unique(f):
        ix = np.flatnonzero(f==outer)
        for a,b in GroupKFold(4,shuffle=True,random_state=542).split(ix,groups=g[ix]):
            t,q = ix[a],ix[b]
            assert not np.isin(g[t],g[q]).any()
            t,q = t[dissent[t]],q[dissent[q]]
            seen[q] += 1
            for key,x in features.items():
                scaler = StandardScaler().fit(x[t])
                w = fit(scaler.transform(x[t]),offset[t],y[t]!=base[t])
                votes[key][q] = offset[q]+scaler.transform(x[q])@w[1:]+w[0]>0
    np.testing.assert_array_equal(seen,dissent.astype(int))
    assert all(np.all(~v|dissent) for v in votes.values())
    return votes,dissent,support


def run():
    raw,y,g,f = data()
    names,h,full = banks(raw,y,f)
    ref = latest_reference(y,g,f,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    _,panels = prediction_sets(raw,y,g,f,rows,of)
    results = {}
    all_votes = []
    for (tag,idx,fold,base,anchor),(six,archive) in zip([
            ('full',np.arange(len(y)),f,ref,full[:,names.index('mitra96')]),
            ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        truth = y[idx]
        pls = cached(TAG,tag,idx,fold,g,y)
        vote,dissent,support = route(base,anchor,pls,six,archive,truth,g[idx],fold)
        all_votes.append(vote)
        consensus = ((np.c_[six,archive]>=.5)==base[:,None]).all(1)
        only = dissent & (support==0)
        scope = dict(challenged=int(dissent.sum()),available_corrections=int((dissent & (base!=truth)).sum()),
                     exclusive=int(only.sum()),exclusive_corrections=int((only & (base!=truth)).sum()))
        policies = {}
        for name,v in vote.items():
            pred = base ^ v
            r = stats(pred,base,truth,g[idx],fold)
            for key,mask in [('exclusive',only),('unanimous',consensus),('protected',(anchor>=.5)!=base)]:
                r[key] = dict(fixed=int((v&mask&(base!=truth)).sum()),harmed=int((v&mask&(base==truth)).sum()))
            if tag=='full':
                unseen = ~np.isin(g[idx],og)
                r['new_orders'] = stats(pred[unseen],base[unseen],truth[unseen],g[idx][unseen],fold[unseen])
            policies[name] = r
        results[tag] = dict(scope=scope,policies=policies)
    assert results['full']['scope']['available_corrections']==375
    repeat = {}
    for name,v in all_votes[0].items():
        twice = v[rows] & all_votes[1][name]
        old_right,new_right = champ==oy,ref[rows]==oy
        repeat[name] = dict(both_flipped=int(twice.sum()),fixed_both=int((twice&~old_right&~new_right).sum()),
                            harmed_both=int((twice&old_right&new_right).sum()),
                            opposite_effect=int((twice&(old_right!=new_right)).sum()))
        assert repeat[name]['both_flipped']==sum(repeat[name][k] for k in ('fixed_both','harmed_both','opposite_effect'))
    results['both_protocols'] = repeat
    out = ROOT/'reports/nov23_matched_learning_curve/broad_error_transport_pls_gate.json'
    out.write_text(json.dumps(results,indent=2)+'\n')
    for name,r in results['full']['policies'].items():
        hist = results['historical']['policies'][name]
        print(name,r['corrected'],r['harmed'],hist['corrected'],hist['harmed'],r['new_orders']['net'],repeat[name])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
