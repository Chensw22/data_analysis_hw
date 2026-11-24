"""Rank all incumbent errors and measure precision at edit counts large enough for the goal."""
import json

import numpy as np
from scipy.linalg import cholesky
from scipy.special import logit
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_solo_coverage_n24 import cached


def scores(x,y,g,fold):
    result = {name:np.full(len(x),np.nan) for name in ('logistic','class_covariance')}
    for f in np.unique(fold):
        rows = np.flatnonzero(fold==f)
        for a,b in GroupKFold(4,shuffle=True,random_state=454).split(rows,groups=g[rows]):
            t,q = rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            scale = StandardScaler().fit(x[t])
            train,ask = [scale.transform(x[ix]).clip(-6,6) for ix in (t,q)]
            model = LogisticRegression(C=.05,max_iter=500).fit(train,y[t])
            result['logistic'][q] = model.decision_function(ask)
            density = []
            for label in (False,True):
                use = y[t]==label
                dist = LedoitWolf().fit(train[use])
                # Score relative likelihood at each query passenger.
                factor = cholesky(dist.covariance_,lower=True)
                logdet = 2*np.log(factor.diagonal()).sum()
                density.append(np.log(use.mean())-.5*(dist.mahalanobis(ask)+logdet))
            result['class_covariance'][q] = density[1]-density[0]
    assert all(np.isfinite(a).all() for a in result.values())
    return result


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    columns,panels = prediction_sets(raw,y,g,fold,rows,of)
    family = np.array([name.split('/')[0] for name in columns])
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[k],v)
        hard = z['remaining']
    report = {}
    for (tag,ix,f,base,p),(six,archive) in zip([
        ('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
        ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        error = y[ix]!=base
        target = 85*len(ix)//100+1
        needed = target-int((~error).sum())
        def against(matrix):
            return np.where(base[:,None],1-matrix,matrix)
        features = [base.astype(float),against(p[:,None]).ravel()]
        for views in [six[:,:3],six[:,3:],*[archive[:,family==k] for k in np.unique(family)],
                      *[cached(k,tag,ix,f,g,y) for k in ('dependency_tree','latent_profiles','broad_error_transport_pair_surprise')]]:
            evidence = logit(against(views).clip(.005,.995))
            features.extend([evidence.mean(1),evidence.max(1)-evidence.min(1)])
        ranking = scores(np.column_stack(features),error,g[ix],f)
        ranking['base'] = against(p[:,None]).ravel()
        agreement = ((six>=.5)==base[:,None]).all(1)
        result = {}
        for name,score in ranking.items():
            cuts = {}
            for fraction in (.04,.10,.20):
                # Allocate the same rate separately in every outer validation roster.
                select = np.zeros(len(ix),bool)
                for level in np.unique(f):
                    ids = np.flatnonzero(f==level)
                    top = ids[np.argsort(-score[ids],kind='stable')[:int(np.ceil(fraction*len(ids)))]]
                    select[top] = True
                edits = int(select.sum())
                fixed = int((select & error).sum())
                minimum = (needed+edits+1)//2
                cuts[str(fraction)] = dict(edits=edits,required_fixes=minimum,fixed=fixed,harmed=edits-fixed,
                    net=2*fixed-edits,precision=fixed/edits,required_precision=minimum/edits,
                    six_fixed=int((select & error & agreement).sum()),hard_fixed=int((select & error & hard[ix]).sum()))
            result[name] = cuts
        report[tag] = dict(rows=len(ix),initial_correct=int((~error).sum()),needed=needed,methods=result)
        print(tag,needed,{name:[(r['fixed'],r['harmed'],r['required_fixes']) for r in cuts.values()]
                          for name,cuts in result.items()},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/wide_error_budget.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
