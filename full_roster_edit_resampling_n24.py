"""Stress-test net-accuracy calibration on new booking samples and inspect its full test decisions."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from export_budget_quarter_n23 import source
from full_roster_booking_validation_n23 import OUT
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_direct_edit_utility_n24 import features,fit
from full_roster_disagreement_calibration_n24 import stats
from full_roster_repartition_gate_n24 import panel_data
from kaggle_workflow import sha256
from sklearn.model_selection import GroupKFold


def calibrate(x,p,y,g,t):
    mean,scale = x[t].mean(0),x[t].std(0).clip(.15)
    train = np.column_stack([np.ones(len(t)),((x[t]-mean)/scale).clip(-5,5)])
    w = fit(train,logit(p[t].clip(.003,.997)),y[t],g[t],False)
    return dict(mean=mean.tolist(),scale=scale.tolist(),w=w.tolist(),rows=len(t))


def shift(model,x):
    w = np.array(model['w'])
    return w[0]+((x-model['mean'])/model['scale']).clip(-5,5)@w[1:]


def subsample(t,groups,seed):
    if seed==0:
        return t
    order = np.unique(groups[t])
    subset = np.random.default_rng(seed).choice(order,int(len(order)*.8),replace=False)
    return t[np.isin(groups[t],subset)]


def inputs():
    raw,y,g,fold,names,h,full,ref,_,cols,rotated = panel_data()
    js = [names.index(k) for k in ('mitra96','v44_without_name','tabicl')]
    half,rot,large = h[:,js],rotated[:,[cols.index(j) for j in js]],full[:,js]
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    sets = [(panel,s[:,0],features(raw,size,s)) for panel,s in ((fold%2,half),((fold//2)%2,rot))]
    return raw,y,g,fold,ref,large[:,0],features(raw,size,large),sets


def test_inputs(train_groups):
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    group = test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(group,train_groups).any()
    mitra,_,mh = source('proposal_online_first_counts15',test)
    incumbent,champion,ch = source('proposal_v131_party_consensus',test)
    manifest = json.loads((ROOT/'reports/proposal_n23_adaptive_bill_prior/manifest.json').read_text())
    parents = dict(v96=mh,v131=ch)
    scores = [mitra]
    for key in ('v44_without_name','tabicl'):
        path = OUT/f'original_{key}.npz'
        parents[key] = sha256(path)
        assert parents[key]==manifest['crossfamily_query_hashes'][key]
        with np.load(path) as z:
            np.testing.assert_array_equal(z['test_ids'],test.PassengerId)
            scores.append(z['test'])
    return test,features(test,test.groupby(group).PassengerId.transform('size').to_numpy(),np.column_stack(scores)),mitra,incumbent,champion,parents


def run():
    raw,y,g,fold,ref,p,ask,sets = inputs()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    size = raw.groupby(g).PassengerId.transform('size').to_numpy()
    historic_x = features(raw.iloc[rows],size[rows],old)
    test,query,mt,_,official,_ = test_inputs(g)
    previous = json.loads((ROOT/'reports/nov23_matched_learning_curve/edit_panel_transfer.json').read_text())
    new = ~np.isin(g,og)
    report = {}
    trials = []
    for seed in (0,1443,1444,1445):
        shifts = np.full((len(y),2),np.nan)
        models = []
        for j,(panel,prob,x) in enumerate(sets):
            for f in np.unique(fold):
                q = np.flatnonzero(fold==f)
                t = np.flatnonzero((panel==panel[q[0]]) & (fold!=f))
                t = subsample(t,g,seed+f+9*j if seed else 0)
                assert not np.isin(g[t],g[q]).any()
                m = calibrate(x,prob,y,g,t)
                models.append(m)
                shifts[q,j] = shift(m,ask[q])
        pred = np.where((p>=.5)==ref,logit(p.clip(.003,.997))+shifts.mean(1)>=0,ref)
        hs = np.full(len(oy),np.nan)
        for f in np.unique(of):
            ix = np.flatnonzero(of==f)
            for a,b in GroupKFold(4,shuffle=True,random_state=633).split(ix,groups=og[ix]):
                t,q = ix[a],ix[b]
                t = t[(old[t,0]>=.5)==champ[t]]
                assert not np.isin(og[t],og[q]).any()
                members = [calibrate(historic_x,old[:,0],oy,og,subsample(t,og,seed+d if seed else 0))
                           for d in (0,57)]
                hs[q] = np.mean([shift(m,historic_x[q]) for m in members],axis=0)
        assert np.isfinite(hs).all() and np.isfinite(shifts).all()
        past = np.where((old[:,0]>=.5)==champ,logit(old[:,0].clip(.003,.997))+hs>=0,champ)
        query_shift = np.mean([shift(m,query) for m in models],axis=0)
        trials.append((shifts.mean(1),hs,query_shift))
        released = np.where((mt>=.5)==official,logit(mt.clip(.003,.997))+query_shift>=0,official)
        res = dict(full=stats(pred,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
                   new_orders=stats(pred[new],ref[new],y[new],g[new],fold[new]),
                   query_changes=int((released!=official).sum()),query_to_true=int((released & (released!=official)).sum()))
        if seed==0:
            for metric in ('corrected','harmed'):
                assert res['full'][metric]==previous['averaged']['net_edits']['full'][metric]
        report[str(seed)] = res
        print(seed,[(k,res[k]['net']) for k in ('full','historical','new_orders')],res['query_changes'],flush=True)
    avg,old_avg,test_avg = [np.mean(v,axis=0) for v in zip(*trials,strict=True)]
    def classify(score,addition,base):
        return np.where((score>=.5)==base,logit(score.clip(.003,.997))+addition>=0,base)
    current = classify(p,avg,ref)
    past = classify(old[:,0],old_avg,champ)
    released = classify(mt,test_avg,official)
    report['bagged'] = dict(full=stats(current,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
        new_orders=stats(current[new],ref[new],y[new],g[new],fold[new]),query_changes=int((released!=official).sum()),
        query_to_true=int((released & (released!=official)).sum()))
    print('bagged', [report['bagged'][k]['net'] for k in ('full','historical','new_orders')],report['bagged']['query_changes'])
    (ROOT/'reports/nov23_matched_learning_curve/edit_resampling.json').write_text(json.dumps(report,indent=2)+'\n')
    return current,past,released


if __name__=='__main__':
    with threadpool_limits(1):
        run()
