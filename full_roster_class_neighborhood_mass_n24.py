"""Estimate relative class coverage with randomly sampled, density-adaptive neighborhoods."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_kernel_repair_overlap_n24 import from_cache
from full_roster_mistake_influence_n24 import design

MODES = ('personal', 'density', 'personal_density', 'personal_shuffled_density')


def distances(x,c):
    return np.maximum((x*x).sum(1)[:,None]+(c*c).sum(1)-2*x@c.T,0.)


def class_profile(x, donors, q, seed):
    rng = np.random.default_rng(seed)
    features = []
    for count in [16,64]:
        draws = []
        for _ in range(24):
            centers = x[rng.choice(donors,count,replace=False)]
            between = distances(centers,centers)
            # Identical measurements belong at the same point in this feature space.
            between[between<1e-9] = np.inf
            radius = np.min(between,axis=1)
            d = distances(x[q],centers)
            nearest = d.argmin(axis=1)
            dist = d[np.arange(len(q)),nearest]
            ratio = dist/radius[nearest].clip(1e-9)
            draws.append(np.column_stack([np.log1p(dist),np.log1p(ratio),ratio<=1]))
        draws = np.stack(draws)
        features.extend([draws.mean(0),np.quantile(draws[:,:,:2],.2,axis=0)])
    return np.column_stack(features)


def predict(x,y,g,t,q,selected=None):
    output = []
    for fit,val in GroupKFold(2,shuffle=True,random_state=267).split(t,groups=g[t]):
        geom,cal = t[fit],t[val]
        assert not np.isin(g[geom],g[cal]).any() and not np.isin(g[t],g[q]).any()
        z = ((x-x[geom].mean(0))/x[geom].std(0).clip(.2)).clip(-7,7)
        ask = np.r_[cal,q]
        candidates = {}
        for kind in ('true','shuffled'):
            target = y[geom] if kind=='true' else np.random.default_rng(919).permutation(y[geom])
            neg,pos = [class_profile(z,geom[target==side],ask,741+int(side)) for side in (False,True)]
            candidates[kind] = np.column_stack([pos,neg,pos-neg])
        matrices = [z[ask],candidates['true'],np.column_stack([z[ask],candidates['true']]),
                    np.column_stack([z[ask],candidates['shuffled']])]
        columns = []
        for j,mat in enumerate(matrices):
            if selected is not None and j not in selected:
                continue
            classifier = make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=350))
            classifier.fit(mat[:len(cal)],y[cal])
            columns.append(classifier.predict_proba(mat[len(cal):])[:,1])
        output.append(np.column_stack(columns))
    return np.mean(output,axis=0)


def protocol(x,y,g,rows,fold,tag,scorer=predict,modes=MODES,experiment='class_neighborhood_mass'):
    folder = ROOT/'artifacts'/f'nov24_{experiment}'
    folder.mkdir(exist_ok=True)
    scores = np.full((len(rows),len(modes)),np.nan)
    for f in np.unique(fold):
        mask = fold==f
        q = rows[mask]
        t = rows[~mask & ~np.isin(g[rows],g[q])]
        path = folder/f'{tag}_{f}.npz'
        if not path.exists():
            pred = scorer(x,y,g,t,q)
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,t=t,q=q,truth=y[t],p=pred)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for k,v in [('t',t),('q',q),('truth',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            scores[mask] = z['p']
        print(tag,int(f),flush=True)
    assert np.isfinite(scores).all()
    return scores


def run(scorer=predict,modes=MODES,experiment='class_neighborhood_mass'):
    _,oy,og,rows,ofold,_,champ,_,_,_,_,old,_ = aligned()
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,half,full)
    gate,_ = gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    all_rows = np.arange(len(y))
    x = design(raw)
    a = protocol(x,y,g,rows,ofold,'historical',scorer,modes,experiment)
    b = protocol(x,y,g,all_rows,fold,'full',scorer,modes,experiment)
    _,rivals = from_cache('full',all_rows,fold,y,g)
    common = ((full>=.5)!=y[:,None]).all(1)
    missed = common & ((rivals>=.5)!=y[:,None]).all(1)
    hard = common[rows] & (champ!=oy)
    new = ~np.isin(g,og)
    assert missed.sum()==243
    report = {}
    for j,mode in enumerate(modes):
        report[mode] = {}
        for w in (1.,.25):
            p = (1-w)*gate+w*b[:,j]>=.5
            prior = (1-w)*old[:,0]+w*a[:,j]>=.5
            if w<1:
                prior = np.where((old[:,0]>=.5)==champ,prior,champ)
            v = dict(full=stats(p,ref,y,g,fold),historical=stats(prior,champ,oy,og,ofold),
                     unseen=stats(p[new],ref[new],y[new],g[new],fold[new]),
                     common_fixed=int((common & (p==y)).sum()),new_on_243=int((missed & (p==y)).sum()),
                     persistent=int((hard & (p[rows]==oy) & (prior==oy)).sum()),
                     fixed_ids=raw.loc[missed & (p==y),'PassengerId'].tolist())
            report[mode][str(w)] = v
            print(mode,w,[v[s]['net'] for s in ['historical','full','unseen']],v['new_on_243'],flush=True)
    path = ROOT/'reports/nov23_matched_learning_curve'/f'{experiment}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
