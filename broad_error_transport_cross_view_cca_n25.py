"""Measure shared registration/spending axes and cross-view reconstruction discrepancies."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_two_view_reliability_n24 import design, tree

MODES = ['raw','shared','residual','raw_residual','supervised_shared','supervised_residual',
         'raw_supervised_residual']
TAG = 'broad_error_transport_cross_view_cca'


def standardize(v, unsup):
    observed = np.isfinite(v)
    support = observed[unsup].sum(0)
    average = np.nansum(v[unsup],axis=0) / support.clip(1)
    imputed = np.where(observed,v,average)
    z = np.column_stack([imputed,~observed])
    scaled = ((z-z[unsup].mean(0))/z[unsup].std(0).clip(.2)).clip(-8.,8.)
    return scaled-scaled[unsup].mean(0)


def whitening(z, unsup):
    cov = z[unsup].T@z[unsup]/len(unsup) + np.eye(z.shape[1])*.1
    eigenvalues,vectors = np.linalg.eigh(cov)
    transform = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    np.testing.assert_allclose(transform@cov@transform,np.eye(len(cov)),atol=1e-8)
    return z@transform


def residual(actual,opposite,unsup):
    design = np.column_stack([np.ones(len(opposite)),opposite])
    v = design[unsup]
    fit = np.linalg.solve(v.T@v+.001*np.eye(v.shape[1]),v.T@actual[unsup])
    return actual-design@fit


def cross_views(views,unsup,t,q,y):
    left,right = [standardize(v,unsup) for v in views]
    zs,zt = [whitening(z,unsup) for z in (left,right)]
    np.testing.assert_allclose(zs[unsup].mean(0),0.,atol=1e-10)
    np.testing.assert_allclose(zt[unsup].mean(0),0.,atol=1e-10)
    covariance = zs[unsup].T@zt[unsup]/len(unsup)
    signs = y[t].astype(float)-y[t].mean()
    # The rank-one term is the between-outcome component of the cross-covariance.
    supervised = np.outer(zs[t].T@signs, zt[t].T@signs)/(len(t)**2*signs.var())
    raw = np.column_stack([left,right])
    query = np.r_[t,q]
    matrices = [raw[query]]
    for cross in (covariance,covariance+supervised):
        u,_,v = np.linalg.svd(cross,full_matrices=False)
        a,b = zs@u[:,:6],zt@v[:6].T
        shared = (a+b)/2
        difference = a-b
        extra = np.column_stack([difference,residual(left,b,unsup),residual(right,a,unsup)])
        matrices.extend([shared[query],extra[query],np.column_stack([raw[query],extra[query]])])
    return matrices


def protocol(views,all_groups,rows,fold,y,tag):
    folder = ROOT/'artifacts'/f'nov25_{TAG}_centered'
    folder.mkdir(exist_ok=True)
    out = np.full((len(rows),len(MODES)),np.nan)
    for f in np.unique(fold):
        mask = fold==f
        q = rows[mask]
        t = rows[~mask & ~np.isin(all_groups[rows],all_groups[q])]
        unsup = np.flatnonzero(~np.isin(all_groups,all_groups[q]))
        assert np.isin(t,unsup).all() and not np.isin(all_groups[t],all_groups[q]).any()
        path = folder/f'{tag}_{f}.npz'
        if not path.exists():
            matrices = cross_views(views,unsup,t,q,y)
            scores = np.column_stack([tree(19).fit(x[:len(t)],y[t]).predict_proba(x[len(t):])[:,1]
                                      for x in matrices])
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,q=q,t=t,truth=y[t],unsup=unsup,p=scores)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as record:
            for key,expected in [('q',q),('t',t),('truth',y[t]),('unsup',unsup)]:
                np.testing.assert_array_equal(record[key],expected)
            out[mask] = record['p']
        print(tag,int(f),flush=True)
    assert np.isfinite(out).all()
    return out


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,b = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,b)
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda col:col!='Name')
    roster = pd.concat([raw,test],ignore_index=True)
    all_groups = roster.PassengerId.str.split('_').str[0].to_numpy()
    np.testing.assert_array_equal(all_groups[:len(y)],g)
    assert not np.isin(all_groups[len(y):],g).any()
    views = design(roster,all_groups)
    out = {}
    for (tag,ix,f,base,p),(six,archive) in zip([
            ('full',np.arange(len(y)),fold,ref,b[:,names.index('mitra96')]),
            ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        truth = y[ix]
        scores = protocol(views,all_groups,ix,f,y,tag)
        consensus = ((np.c_[six,archive]>=.5)==base[:,None]).all(1)
        if tag=='full':
            assert (consensus & (truth!=base)).sum()==240
        summary = {}
        for j,key in enumerate(MODES):
            for name,w in [('direct',1.),('quarter',.25)]:
                def decide(score):
                    blended = (1-w)*p+w*score >= .5
                    return blended if w==1 else np.where((p>=.5)==base,blended,base)
                pred,control = decide(scores[:,j]),decide(scores[:,0])
                result = stats(pred,base,truth,g[ix],f)
                result['versus_raw'] = stats(pred,control,truth,g[ix],f)
                for subgroup,select in [('unanimous',consensus),('disagreement',~consensus)]:
                    result[subgroup] = dict(corrected=int((select & (truth!=base) & (pred==truth)).sum()),
                                            harmed=int((select & (truth==base) & (pred!=truth)).sum()))
                if tag=='full':
                    new = ~np.isin(g[ix],og)
                    result['new_orders'] = stats(pred[new],base[new],truth[new],g[ix][new],f[new])
                summary[key+'_'+name] = result
                print(tag,key,name,result['corrected'],result['harmed'],result['versus_raw']['net'],
                      result['unanimous']['corrected'],flush=True)
        out[tag] = summary
    (ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
