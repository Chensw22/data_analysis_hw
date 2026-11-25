"""Measure whether knowing which archive treatments disagree improves an incumbent probability."""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from adjudicate_amount_dissent_n25 import counts
from disjoint_tree_fields_n25 import ART, DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from grouped_peer_teacher_n25 import OUT as REFERENCE


def panels_and_reference():
    raw,y,g,fold=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows])
        np.testing.assert_array_equal(z['y'],y[rows])
        old_native=z['scores'][:,4]
    with np.load(ART/'v96_oof_completion/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'],rows)
        np.testing.assert_array_equal(z['prob'],old_native)
    names,panels=prediction_sets(raw,y,g,fold,rows,hf)
    names=[*[f'response_channels/{n}' for n in ('additive','any_event','odd_events')],
           *[f'broad_error_transport_projection/{n}' for n in ('linear','fixed_curves','learned_curves')],*names]
    p,h=(np.column_stack(panel) for panel in panels)
    assert len(names)==p.shape[1]==h.shape[1]==68 and len(set(names))==68
    native=np.empty(len(y))
    reference=np.empty((3,len(y)),int)
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        with np.load(REFERENCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['y'],y[q])
            native[q]=z['native']
            reference[:,q]=[z[n] for n in ('v131','nine_on_low_branch','nine_on_high_branch')]
    return y,g,fold,rows,names,p,h,native,old_native,reference


def representations(p,native,names):
    odds=logit(np.clip(p,.005,.995))
    anchor=logit(np.clip(native,.002,.998))
    family=np.array([n.split('/')[0] for n in names])
    center=np.mean(odds,axis=1)
    medians=[]
    deviations=[]
    ordered=[]
    for name in dict.fromkeys(family):
        members=odds[:,family==name]
        middle=np.median(members,axis=1)
        medians.append(middle-center)
        differences=members-middle[:,None]
        if members.shape[1]>1:
            deviations.append(differences)
            ordered.append(np.sort(differences,axis=1))
    # Identity-blind controls still see strength and spread, including the same multiset of deviations.
    basic=np.column_stack([np.ones(len(p)),anchor,center-anchor,np.quantile(odds,[.1,.9],axis=1).T-center[:,None]])
    family_features=np.column_stack([basic,*medians])
    return dict(distribution=basic,family=family_features,
                anonymous_treatments=np.column_stack([family_features,*ordered]),
                named_treatments=np.column_stack([family_features,*deviations]))


def fit(x,offset,y):
    def objective(beta):
        score=offset+x@beta
        loss=np.logaddexp(0.,score)-y*score
        # Fixed Gaussian shrinkage on corrections to the existing log-odds.
        return loss.sum()+8*np.dot(beta,beta), x.T@(expit(score)-y)+16*beta
    result=minimize(objective,np.zeros(x.shape[1]),method='L-BFGS-B',jac=True,
                    options={'maxiter':400,'ftol':1e-10})
    assert result.success,result.message
    return result.x


def run():
    y,g,f,rows,names,p,h,native,old_native,reference=panels_and_reference()
    new=~np.isin(g,g[rows])
    full=np.arange(len(y))
    design=representations(np.vstack([p,h]),np.r_[native,old_native],names)
    target=np.r_[y,y[rows]]
    base=logit(np.r_[native,old_native].clip(.002,.998))
    ids=np.r_[full,rows]
    protocols={'same_model_all':[],'same_model_new':[],'historical_to_new':[(len(y)+np.arange(len(rows)),full[new])]}
    for k in np.unique(f):
        q=full[f==k]
        for a,b in GroupKFold(4,shuffle=True,random_state=2621).split(q,groups=g[q]):
            protocols['same_model_all'].append((q[a],q[b]))
        protocols['same_model_new'].append((q[~new[q]],q[new[q]]))
    votes=(p>=.5).sum(1)
    old_votes=(h>=.5).sum(1)
    minor=np.minimum(votes,68-votes)
    repeated=np.zeros(len(y),bool)
    repeated[rows]=(minor[rows]==0)&((old_votes==0)|(old_votes==68))&((old_votes>34)==(votes[rows]>34))
    actual=native>=.5
    prior,lo,hi=reference
    known=prior>=0
    report={}
    storage={}
    for protocol,splits in protocols.items():
        report[protocol]={}
        for name,matrix in design.items():
            prob=np.full(len(y),np.nan)
            for train,q in splits:
                assert not np.isin(g[ids[train]],g[q]).any() and np.isnan(prob[q]).all()
                beta=fit(matrix[train],base[train],target[train])
                prob[q]=expit(base[q]+matrix[q]@beta)
            ix=np.flatnonzero(np.isfinite(prob))
            np.testing.assert_array_equal(ix,full if protocol=='same_model_all' else full[new])
            storage[protocol+'_'+name]=prob
            a,b=edit_effect_bounds(prob[ix]>=.5,actual[ix],prior[ix],lo[ix],hi[ix],y[ix])
            np.testing.assert_array_equal(a[known[ix]],b[known[ix]])
            result={'all_net_bounds':[int(a.sum()),int(b.sum())]}
            for label,mask in [('known',known),('unanimous',known&(minor==0)),('repeated',known&repeated),
                               ('one_to_three',known&(minor>=1)&(minor<=3)),('four_plus',known&(minor>=4))]:
                result[label]=counts(a,mask[ix],f[ix])
            report[protocol][name]=result
            print(protocol,name,result['all_net_bounds'],[(result[t]['fixed'],result[t]['harmed']) for t in ('known','unanimous','four_plus')])
    output=ART/'nov26_family_identity'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/'probabilities.npz',fold=f,rows=rows,**storage)
    DEST.with_name('family_identity_transfer.json').write_text(json.dumps(dict(columns=names,results=report),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
