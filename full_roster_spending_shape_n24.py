"""Learn directions of positive charge effects separately for single-service and mixed baskets."""
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,BILLS,banks,data,latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_two_view_reliability_n24 import design

MODES = ('free','pooled_direction','basket_direction','reversed_direction')


def directions(raw,fees,t,y):
    paid = fees>0
    counts = paid.sum(1)
    mask_codes = paid.astype(int) @ (1<<np.arange(len(BILLS)))
    planet = raw.HomePlanet.fillna('?')
    sleep = raw.CryoSleep.astype(str)
    result = np.zeros((3,len(BILLS)),int)
    support = np.zeros_like(result)
    for j in range(len(BILLS)):
        other = np.nansum(np.delete(fees,j,axis=1),axis=1)
        groups = pd.MultiIndex.from_arrays([planet,sleep,mask_codes,np.digitize(other,[1,300,1000,3000])]).factorize()[0]
        for row,active in enumerate([paid[:,j],paid[:,j]&(counts==1),paid[:,j]&(counts>1)]):
            idx = t[active[t]]
            covariance = 0.
            for k in np.unique(groups[idx]):
                members = idx[groups[idx]==k]
                if len(members)<4 or y[members].min()==y[members].max():
                    continue
                ranks = rankdata(fees[members,j])
                if ranks.max()==ranks.min():
                    continue
                covariance += ((ranks-ranks.mean())/len(ranks)) @ y[members].astype(float)
                support[row,j] += len(members)
            result[row,j] = np.sign(covariance) if support[row,j]>=30 else result[0,j]
    return result,support


def inputs(raw,g):
    reg,_ = design(raw,g)
    fees = raw[BILLS].to_numpy(float)
    paid = fees>0
    amounts = np.nan_to_num(np.log1p(fees))
    count = paid.sum(1)
    common = np.column_stack([reg,raw.CryoSleep.eq(True),raw.CryoSleep.isna(),paid,np.isnan(fees)])
    x = np.column_stack([common,amounts*(count==1)[:,None],amounts*(count>1)[:,None]])
    return x,fees,common.shape[1]


def protocol(raw,y,g,rows,fold,tag):
    x,fees,width = inputs(raw,g)
    scores = np.full((len(rows),len(MODES)),np.nan)
    audit = []
    folder = ROOT/'artifacts/nov24_spending_shape'
    folder.mkdir(exist_ok=True)
    for f in np.unique(fold):
        selected = fold==f
        q = rows[selected]
        t = rows[~selected & ~np.isin(g[rows],g[q])]
        assert not np.isin(g[t],g[q]).any()
        direction,support = directions(raw,fees,t,y)
        path = folder/f'{tag}_{f}.npz'
        if not path.exists():
            estimates = []
            for rule in (np.zeros(10,int),np.tile(direction[0],2),direction[1:].ravel(),-direction[1:].ravel()):
                constraints = np.r_[np.zeros(width,int),rule].tolist()
                model = LGBMClassifier(n_estimators=300,num_leaves=17,min_child_samples=30,learning_rate=.045,
                          n_jobs=2,random_state=889,verbosity=-1,monotone_constraints=constraints,
                          monotone_constraints_method='advanced')
                model.fit(x[t],y[t])
                estimates.append(model.booster_.predict(x[q],num_threads=2))
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=t,q=q,y=y[t],direction=direction,p=np.column_stack(estimates))
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,v in [('t',t),('q',q),('y',y[t]),('direction',direction)]:
                np.testing.assert_array_equal(z[k],v)
            scores[selected] = z['p']
        audit.append(dict(fold=int(f),directions=direction.tolist(),support=support.tolist()))
    assert np.isfinite(scores).all()
    return scores,audit


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(ref,gate>=.5)
    a,old_dirs = protocol(raw,y,g,rows,of,'historical')
    b,new_dirs = protocol(raw,y,g,np.arange(len(y)),fold,'full')
    unseen = ~np.isin(g,og)
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    report = dict(services=BILLS, historical_directions=old_dirs,full_directions=new_dirs)
    for j,name in enumerate(MODES):
        report[name] = {}
        for w in (1.,.25):
            p = (w*b[:,j]+(1-w)*gate)>=.5
            past = (w*a[:,j]+(1-w)*old[:,0])>=.5
            if w<1:
                past = np.where((old[:,0]>=.5)==champ,past,champ)
            fixed = hard & (p==y)
            r = dict(full=stats(p,ref,y,g,fold),historical=stats(past,champ,oy,og,of),
                     unseen=stats(p[unseen],ref[unseen],y[unseen],g[unseen],fold[unseen]),
                     fixed_among_147=int(fixed.sum()),repeated=int((fixed[rows] & (past==oy) & (champ!=oy)).sum()),
                     fixed_ids=raw.loc[fixed,'PassengerId'].tolist())
            report[name][str(w)] = r
            print(name,w,[r[k]['net'] for k in ('full','historical','unseen')],fixed.sum(),flush=True)
        report[name]['versus_free'] = stats(b[:,j]>=.5,b[:,0]>=.5,y,g,fold)
    path = ROOT/'reports/nov23_matched_learning_curve/spending_shape.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
