"""Test known correct and incorrect reference bookings as evidence for new common errors."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conflict_pair_ordering_n26 import CACHE,REPORT as RANK_REPORT
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,data
from kaggle_workflow import sha256
from outside_panel_coverage_n26 import archives


def retrieve(queries,bank,edge,dist,x,side,marked,groups):
    in_bank = np.zeros(len(x),bool)
    in_bank[bank] = True
    features = np.zeros((len(queries),15))
    supported = np.zeros(len(queries),bool)
    for k,i in enumerate(queries):
        refs = edge[i]
        yes = in_bank[refs]&(side[i]==side[refs])
        refs = refs[yes]
        physical = dist[i][yes]
        # Two metrics from the raw observations and the independently fitted models.
        # Use each donor booking at most once per class in each comparison.
        model_gap = np.sqrt(np.mean((x[refs]-x[i])**2,axis=1))
        for status in (0,1):
            keep = marked[refs]==status
            pos = np.flatnonzero(keep)
            if not len(pos):
                continue
            features[k,7*status] = np.log1p(len(np.unique(groups[refs[pos]])))
            for metric,j in [(physical,1),(model_gap,4)]:
                selected = pos[np.lexsort((refs[pos],metric[pos]))]
                _,idx = np.unique(groups[refs[selected]],return_index=True)
                nearest = selected[np.sort(idx)[:3]]
                features[k,7*status+j:7*status+j+3] = [physical[nearest].mean(),model_gap[nearest].mean(),
                                                          x[refs[nearest],0].mean()-x[i,0]]
        supported[k] = features[k,0]>0 and features[k,7]>0
        features[k,-1] = supported[k]
    return features,supported


def shuffle_marks(bank,marks,risk,side,regions,seed):
    fake = marks.copy()
    rng = np.random.default_rng(seed)
    for level in np.unique(regions[bank]):
        for direction in (False,True):
            members = bank[(regions[bank]==level)&(side[bank]==direction)]
            members = members[np.argsort(risk[members],kind='stable')]
            for cell in np.array_split(members,max(1,len(members)//12)):
                fake[cell] = rng.permutation(marks[cell])
    return fake


def predict_model(train,query,truth):
    norm = StandardScaler().fit(train)
    fit = LogisticRegression(C=.045,max_iter=450).fit(norm.transform(train).clip(-6,6),truth)
    return fit.predict_proba(norm.transform(query).clip(-6,6))[:,1]


def inputs_for_banks(raw,y,g,f):
    archive,_ = archives(raw,y,g,f,True)
    names = ['Mitra2_adapted','TabDPT','LimiX2','mask_net_joint','conditional_bill_flow','latent_class_mixture']
    p = np.column_stack([archive[n] for n in names])
    path = CACHE/'held.npz'
    assert sha256(path)==json.loads(RANK_REPORT.read_text())['cached_predictions_sha256']
    with np.load(path) as z:
        for key,expected in [('passenger',raw.PassengerId.to_numpy(str)),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[key],expected)
        ranks = z['scores']
    side = p[:,0]>=.5
    direction = np.where(side,-1,1)
    values = np.column_stack([logit(p.clip(.003,.997))*direction[:,None],ranks*direction[:,None]])
    marks = side!=y
    a,b,d = relations(raw,g,False)
    ok = (f[a]==f[b])&(side[a]==side[b])
    a,b,d = a[ok],b[ok],d[ok]
    idx = np.argsort(np.r_[a,b],kind='stable')
    splits = np.cumsum(np.bincount(np.r_[a,b],minlength=len(y)))[:-1]
    edges = np.split(np.r_[b,a][idx],splits)
    dist = np.split(np.r_[d,d][idx],splits)
    regions = (raw.HomePlanet.fillna('?')+'_'+raw.CryoSleep.astype('string').fillna('?')).to_numpy(str)
    return values,side,marks,edges,dist,regions


def fit_banks(values,side,marks,edges,dist,regions,g,f,shard_seed):
    scored = np.full((len(g),3,3),np.nan)
    support = np.zeros((len(g),3),bool)
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        shards = [rows[q] for _,q in GroupKFold(4,shuffle=True,random_state=shard_seed).split(rows,groups=g[rows])]
        for j,q in enumerate(shards):
            for repeat,cal in enumerate([h for h in range(4) if h!=j]):
                t = shards[cal]
                bank = np.concatenate([shards[h] for h in range(4) if h not in (cal,j)])
                assert not np.isin(g[q],g[np.r_[bank,t]]).any() and not np.isin(g[t],g[bank]).any()
                scaled = StandardScaler().fit(values[bank]).transform(values).clip(-6,6)
                ids = np.r_[t,q]
                base = np.column_stack([scaled[ids],side[ids]])
                scored[q,repeat,0] = predict_model(base[:len(t)],base[len(t):],marks[t])
                fake = shuffle_marks(bank,marks,values[:,0],side,regions,54+k*19+j*4+cal)
                for c,labels in enumerate((marks,fake),1):
                    info,present = retrieve(ids,bank,edges,dist,scaled,side,labels,g)
                    design = np.column_stack([base,info])
                    scored[q,repeat,c] = predict_model(design[:len(t)],design[len(t):],marks[t])
                    if c==1:
                        support[q,repeat] = present[len(t):]
    assert np.isfinite(scored).all()
    return scored,support


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    values,side,marks,edges,dist,regions = inputs_for_banks(raw,y,g,f)
    scored,support = fit_banks(values,side,marks,edges,dist,regions,g,f,214)
    prob = scored.mean(1)
    base,model,placebo = (prob>.5).T
    actual = (side==cur['parent'])&cur['known']
    actions = dict(individual=base,retrieved=model,shuffled=placebo,new_retrieved=model&~base,
                   only_real=model&~base&~placebo,all_banks=(scored[:,:,1]>.5).all(1),
                   supported=model&support.all(1))
    report = {}
    for scope,mask in [('all',actual),('eleven_agree',actual&cur['common']),('new_orders',actual&cur['new'])]:
        report[scope] = {n:summary(v,marks,mask,f) for n,v in actions.items()}
        report[scope]['support'] = dict(covered_in_any=int((mask&support.any(1)).sum()),
                                        covered_in_all=int((mask&support.all(1)).sum()))
        report[scope]['brier_score'] = dict(zip(('individual','retrieved','shuffled'),
                np.mean((prob[mask]-marks[mask,None])**2,axis=0).tolist(),strict=True))
        print(scope,[(n,report[scope][n]['fixed'],report[scope][n]['harmed']) for n in actions],flush=True)
    report['scope'] = ('Each reference label belongs to a different training order in the same base-model fold. '
       'For each query, three rotating libraries exclude both the query and the calibration bookings. '
       'Missing class-specific donors remain explicit. The shuffled control permutes reference errors '
       'within direction, origin, sleep, and similarly scored bank passengers.')
    (ROOT/'reports/nov23_matched_learning_curve/conflict_case_control_banks.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
