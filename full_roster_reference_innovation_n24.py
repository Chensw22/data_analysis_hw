"""Align new reference models and test the component of their scores unexplained by baseline risk."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from deployment_booking_mix_n23 import passenger_vectors
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_contrast_pairs_n24 import predict as sparse_pair
from full_roster_disagreement_calibration_n24 import stats
from full_roster_kernel_repair_overlap_n24 import from_cache
from full_roster_same_model_calibration_n23 import learn


def bank(raw,y,g,rows,fold,tag):
    paired,_ = sparse_pair(passenger_vectors(raw),y,g,rows,fold,tag)
    columns = np.column_stack([paired[:,1],np.full((len(rows),2),np.nan)])
    for f in np.unique(fold):
        held = fold==f
        q = rows[held]
        t = rows[~held & ~np.isin(g[rows],g[q])]
        for j,(name,col) in enumerate([('class_neighborhood_mass',2),('labelled_reference_pairs',3)],start=1):
            with np.load(ROOT/f'artifacts/nov24_{name}/{tag}_{f}.npz') as z:
                for key,expected in [('q',q),('t',t),('truth',y[t])]:
                    np.testing.assert_array_equal(z[key],expected)
                columns[held,j] = z['p'][:,col]
    assert np.isfinite(columns).all()
    return columns


def policies(p,ref,opinions,y,g,fold):
    # Protected overrides are only applied at query time.
    native = p>=.5
    free = native==ref
    odds = logit(p.clip(.0001,.9999))
    z = logit(opinions.clip(.0001,.9999))
    x = np.column_stack([np.ones(len(y)),odds,*[np.maximum(odds-v,0) for v in (-2.,0.,2.)]])
    reverse = ((opinions>=.5)!=native[:,None]).sum(1)
    rules = {f'votes_{n}':ref ^ (free & (reverse>=n)) for n in (1,2,3)}
    for key in ('base_calibration','new_information','both'):
        rules[key] = ref.copy()
    for f in np.unique(fold):
        outer = np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=915).split(outer,groups=g[outer]):
            t,q = outer[a],outer[b]
            assert not np.isin(g[t],g[q]).any()
            reg = Ridge(alpha=5,fit_intercept=False).fit(x[t],z[t])
            residual = z-reg.predict(x)
            matrices = {'base_calibration':x,'new_information':residual,'both':np.column_stack([x,residual])}
            for key,mat in matrices.items():
                w = learn(mat[t],y[t],odds[t],np.zeros(mat.shape[1]))
                chosen = (odds[q]+mat[q]@w)>=0
                rules[key][q] = np.where(free[q],chosen,ref[q])
    return rules


def run():
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    ix = np.arange(len(y))
    a,b = bank(raw,y,g,rows,ofold,'historical'),bank(raw,y,g,ix,fold,'full')
    _,prev = from_cache('full',ix,fold,y,g)
    shared = ((full>=.5)!=y[:,None]).all(1)
    missed = shared & ((prev>=.5)!=y[:,None]).all(1)
    subset = shared[rows] & (champ!=oy)
    new = ~np.isin(g,og)
    assert missed.sum()==243
    newhit,oldhit = (b>=.5)==y[:,None],(a>=.5)==oy[:,None]
    counts = newhit.sum(1)
    report = dict(in_243_by_votes=np.bincount(counts[missed],minlength=4).tolist(),
                  new_243_by_model=newhit[missed].sum(0).tolist(),
                  repeated_same_method_in_201=int((subset & (newhit[rows] & oldhit).any(1)).sum()),
                  reachable_baseline_errors=int(((ref!=y) & (counts>0)).sum()),
                  correct_baseline_with_any_wrong=int(((ref==y) & (counts<3)).sum()),rules={})
    # At inference only agreement is observable. Include originally correct unanimous decisions
    # when measuring the precision of each specialist's dissent.
    prior_votes = np.column_stack([full,prev])>=.5
    unanimous = prior_votes.all(1) | (~prior_votes).all(1)
    dissent = (b>=.5)!=prior_votes[:,:1]
    alone = unanimous & (dissent.sum(1)==1)
    assert np.array_equal(missed,unanimous & (prior_votes[:,0]!=y))
    report['sole_dissent_in_unanimous_cases'] = {}
    for j,label in enumerate(('sparse_pair','class_coverage','reference_network')):
        cuts = {}
        for direction in [False,True]:
            m = alone & dissent[:,j] & ((b[:,j]>=.5)==direction)
            cuts[str(direction)] = dict(proposed=int(m.sum()),correct=int((m & (y==direction)).sum()))
        report['sole_dissent_in_unanimous_cases'][label] = cuts
    orig = policies(old[:,0],champ,a,oy,og,ofold)
    curr = policies(full[:,names.index('mitra96')],ref,b,y,g,fold)
    for key,p in curr.items():
        v = dict(historical=stats(orig[key],champ,oy,og,ofold),full=stats(p,ref,y,g,fold),
                 unseen=stats(p[new],ref[new],y[new],g[new],fold[new]),new_on_243=int((missed & (p==y)).sum()),
                 repeated_in_201=int((subset & (p[rows]==oy) & (orig[key]==oy)).sum()))
        report['rules'][key] = v
        print(key,[v[k]['net'] for k in ('historical','full','unseen')],v['new_on_243'],flush=True)
    print(report['in_243_by_votes'],report['new_243_by_model'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve/reference_innovation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
