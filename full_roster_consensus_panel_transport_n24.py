"""Train on honest half-roster base-model errors and reweight toward unanimous query passengers."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT
from full_roster_mistake_influence_n24 import design
from full_roster_repartition_gate_n24 import panel_data
from full_roster_six_failure_archive_n24 import prediction_sets


def representation(p,raw):
    base = p[:,4]>=.5
    against = np.where(base[:,None],1-p,p)
    scores = logit(against.clip(.003,.997))
    count = ((p>=.5)!=base[:,None]).sum(1)
    return np.column_stack([base,scores,scores.mean(1),count,raw])


def transport(x,z):
    # Equal-prior source-vs-query discrimination gives the density ratio as odds.
    both = np.vstack([x,z])
    domain = np.r_[np.zeros(len(x)),np.ones(len(z))]
    w = np.r_[np.ones(len(x)),np.full(len(z),len(x)/len(z))]
    model = LogisticRegression(C=.03,max_iter=500).fit(both,domain,sample_weight=w)
    odds = np.exp(model.decision_function(x).clip(-3,3))
    return odds/odds.mean()


def train(x,z,error,w):
    model = LogisticRegression(C=.05,max_iter=500).fit(x,error,sample_weight=w)
    return model.predict_proba(z)[:,1]


def run():
    raw,y,g,fold,names,orig,full,ref,_,cols,rot = panel_data()
    _,_,old_groups,rows,of,*_ = aligned()
    _,panels = prediction_sets(raw,y,g,fold,rows,of)
    six,archive = panels[0]
    strict = ((np.c_[six,archive]>=.5)==ref[:,None]).all(1)
    assert (strict & (ref!=y)).sum()==240
    v = full[:,cols]
    free = (v[:,4]>=.5)==ref
    x = design(raw)
    query = representation(v,x)
    scores = {}
    supports = []
    for variant,half,panel in [('original',orig[:,cols],fold%2),('rotated',rot,(fold//2)%2)]:
        source = representation(half,x)
        for f in np.unique(fold):
            q = np.flatnonzero((fold==f)&free)
            t = np.flatnonzero((panel==panel[q[0]]) & (fold!=f))
            assert not np.isin(g[t],g[q]).any() and (panel[q]==panel[q[0]]).all()
            error = (half[t,4]>=.5)!=y[t]
            scale = StandardScaler().fit(source[t])
            a = scale.transform(source[t]).clip(-6,6)
            for scope,ids in [('all',q),('unanimous',q[strict[q]])]:
                b = scale.transform(query[ids]).clip(-6,6)
                near = (((half[t]>=.5)!=(half[t,4,None]>=.5)).sum(1)<=2)
                weights = transport(a,b)
                supports.append(dict(variant=variant,fold=int(f),scope=scope,source=len(t),
                    source_errors=int(error.sum()),query=len(ids),effective_sample_size=float(weights.sum()**2/(weights@weights))))
                for mode,s,w in [('natural',np.ones(len(t),bool),np.ones(len(t))),
                                 ('near',near,np.ones(len(t))),('weighted',np.ones(len(t),bool),weights)]:
                    risk = scores.setdefault((variant,scope,mode),np.full(len(y),np.nan))
                    risk[ids] = train(a[s],b,error[s],w[s])
    report = {'source_support':supports,'methods':{}}
    for scope,eligible in [('all',free),('unanimous',strict&free)]:
        for kind in ('natural','near','weighted'):
            a,b = [scores[(variant,scope,kind)] for variant in ('original','rotated')]
            for tag,p in [('original',a),('rotated',b),('average',(a+b)/2)]:
                assert np.array_equal(np.isfinite(p),eligible)
                picked = p>.5
                top = np.zeros(len(y),bool)
                for f in np.unique(fold):
                    ids = np.flatnonzero((fold==f)&strict&eligible)
                    top[ids[np.argsort(-p[ids],kind='stable')[:int(np.ceil(.1*len(ids)))]]] = True
                e = ref!=y
                r = dict(fixed=int((picked&e).sum()),harmed=int((picked&~e).sum()),
                         strict_fixed=int((strict&picked&e).sum()),strict_harmed=int((strict&picked&~e).sum()),
                         top_fixed=int((top&e).sum()),top_harmed=int((top&~e).sum()),
                         strict_ap=float(average_precision_score(e[strict&eligible],p[strict&eligible])))
                new = ~np.isin(g,old_groups)
                r['new_net'] = int((new&picked&e).sum()-(new&picked&~e).sum())
                report['methods'][scope+'/'+kind+'/'+tag] = r
                if tag=='average':
                    print(scope,kind,r,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/consensus_panel_transport.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
