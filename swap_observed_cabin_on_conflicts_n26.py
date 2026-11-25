"""Swap entire observed cabin records within held conflicts, then test factual cabin contrasts."""
import json
from itertools import combinations_with_replacement

import numpy as np
from catboost import CatBoostClassifier,CatBoostRanker,Pool
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from conflict_field_blocks_n26 import designs
from conflict_pair_ordering_n26 import CACHE,REPORT as RANK_REPORT,training_pairs
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import OUT,ROOT,data
from kaggle_workflow import sha256


def analogues(a,b,d,eligible,g):
    left,right = np.r_[a,b],np.r_[b,a]
    dist = np.r_[d,d]
    keep = ~eligible[left]&eligible[right]
    left,right,dist = left[keep],right[keep],dist[keep]
    order = np.lexsort((right,dist,left))
    rows,donors,last,seen = [],[],-1,set()
    for i,j in zip(left[order],right[order],strict=True):
        if i!=last:
            seen = set()
            last = i
        if len(seen)<7 and g[j] not in seen:
            seen.add(g[j])
            rows.append(i)
            donors.append(j)
    return np.array(rows,int),np.array(donors,int)


def model_scores(model,x):
    if isinstance(model,CatBoostClassifier):
        return model.predict(x,prediction_type='RawFormulaVal')
    return model.predict(x)


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    matrices = designs(raw,g)
    x = matrices[3]
    cabin = slice(matrices[1].shape[1],x.shape[1])
    del matrices
    cache = CACHE/'held.npz'
    assert sha256(cache)==json.loads(RANK_REPORT.read_text())['cached_predictions_sha256']
    with np.load(cache) as file:
        np.testing.assert_array_equal(file['passenger'],raw.PassengerId.to_numpy(str))
        pairs = {n:file[n+'_pairs'] for n in ('spending','registration')}
    delta = {name:np.full((len(e),2),np.nan) for name,e in pairs.items()}
    a,b,d = relations(raw,g,False)
    signal = np.full((len(y),2,5),np.nan)
    for k,h in combinations_with_replacement(np.unique(f),2):
        eligible = (f!=k)&(f!=h)
        t = np.flatnonzero(eligible)
        selection = {n:(np.sort(f[e],axis=1)==[k,h]).all(1) for n,e in pairs.items()}
        edges = np.concatenate([p[selection[n]] for n,p in pairs.items()])
        if not len(edges) and k!=h:
            continue
        assert not np.isin(g[edges],g[t]).any()
        real,_ = training_pairs(a,b,d,y,g,eligible)
        params = dict(depth=4,iterations=115,learning_rate=.07,l2_leaf_reg=8,thread_count=2,
                      verbose=False,allow_writing_files=False,random_seed=3505)
        rank = CatBoostRanker(loss_function='PairLogit',**params).fit(Pool(x[t],label=y[t].astype(int),pairs=real))
        for name,p in pairs.items():
            subset = selection[name]
            i,j = p[subset].T
            if len(i):
                actual = rank.predict(x[i])-rank.predict(x[j])
                left,right = x[i].copy(),x[j].copy()
                left[:,cabin],right[:,cabin] = x[j,cabin],x[i,cabin]
                delta[name][subset] = np.column_stack([actual,rank.predict(left)-rank.predict(right)])
        if k!=h:
            continue
        q = np.flatnonzero(f==k)
        i,j = analogues(a,b,d,eligible,g)
        assert np.isin(i,q).all() and np.isin(j,t).all()
        hypothetical = x[i].copy()
        hypothetical[:,cabin] = x[j,cabin]
        point = CatBoostClassifier(**params).fit(x[t],y[t].astype(int))
        count = np.bincount(i,minlength=len(y))
        for m,model in enumerate((rank,point)):
            actual = model_scores(model,x[q])
            alternative = model_scores(model,hypothetical)
            mean = np.bincount(i,weights=alternative,minlength=len(y))[q]/np.maximum(1,count[q])
            mean_sq = np.bincount(i,weights=alternative**2,minlength=len(y))[q]/np.maximum(1,count[q])
            factual = np.zeros(len(y))
            factual[q] = actual
            percentile = np.bincount(i,weights=(alternative<factual[i])+.5*(alternative==factual[i]),minlength=len(y))[q]/np.maximum(1,count[q])
            supported = count[q]>0
            signal[q,m] = np.column_stack([actual,np.where(supported,actual-mean,0),
                      np.sqrt(np.maximum(0,mean_sq-mean**2)),np.where(supported,percentile-.5,0),count[q]/7])
    assert np.isfinite(signal).all()
    report = {'ordering':{},'deployment':{}}
    pieces = raw.Cabin.str.split('/',expand=True)
    key = (pieces[0]+'_'+pieces[2]).fillna('?').to_numpy()
    for n,p in pairs.items():
        actual,swapped = delta[n].T
        assert np.isfinite(actual+swapped).all()
        same = key[p[:,0]]==key[p[:,1]]
        report['ordering'][n] = {}
        for s,sub in [('all',np.ones(len(same),bool)),('same_deck_and_side',same),('different_track',~same)]:
            u,v = actual[sub],swapped[sub]
            report['ordering'][n][s] = dict(pairs=len(u),actual=float(((u>0)+.5*(u==0)).sum()),
                       swapped=float(((v>0)+.5*(v==0)).sum()),symmetrized=float((((u+v)>0)+.5*((u+v)==0)).sum()),
                       actual_correct_swapped_wrong=int(((u>0)&(v<0)).sum()),swapped_correct_actual_wrong=int(((v>0)&(u<0)).sum()))
    prev = json.loads((RANK_REPORT.parent/'conflict_field_blocks.json').read_text())
    for name in pairs:
        assert report['ordering'][name]['all']['actual']==prev['ordering'][name]['correct_equivalents_per_mask'][3]
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        strong = logit(z['p'].clip(.003,.997))
    plain = np.column_stack([strong,signal[:,:,0]])
    enriched = np.column_stack([plain,signal[:,:,1:].reshape(len(y),-1)])
    predictions = np.zeros((len(y),2),bool)
    for k in np.unique(f):
        rows = np.flatnonzero(f==k)
        for u,v in GroupKFold(4,shuffle=True,random_state=169).split(rows,groups=g[rows]):
            t,q = rows[u],rows[v]
            assert not np.isin(g[t],g[q]).any()
            for j,fields in enumerate((plain,enriched)):
                scale = StandardScaler().fit(fields[t])
                fit = LogisticRegression(C=.2,max_iter=400).fit(scale.transform(fields[t]).clip(-6,6),y[t])
                predictions[q,j] = fit.predict(scale.transform(fields[q]).clip(-6,6))
    old,new = predictions.T
    for n,mask in [('all',cur['known']),('common',cur['common']),('new_orders',cur['known']&cur['new'])]:
        extra = (new!=cur['parent'])&(old==cur['parent'])
        report['deployment'][n] = dict(covered_by_other_cabins=int((mask&(signal[:,0,4]>0)).sum()),
                 net_relative_to_control=summary(old!=new,old!=y,mask,f),extra_challenges=summary(extra,cur['wrong'],mask,f))
    report['scope'] = ('Only the complete cabin triple is swapped. Ranking on a held conflict excludes both '
       'orders during training. Deployment contrasts each actual cabin with up to seven other training bookings '
       'with similar observed spending and registration, without consulting the donors’ labels. Source controls '
       'and factual contrasts are calibrated on separate orders inside each base-model fold.')
    (ROOT/'reports/nov23_matched_learning_curve/swap_observed_cabin_on_conflicts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['ordering'],flush=True)
    print({k:v['extra_challenges']['net'] for k,v in report['deployment'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
