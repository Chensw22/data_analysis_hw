"""Train error detectors with direction- and confidence-matched case/control sampling."""
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_mistake_influence_n24 import design
from full_roster_six_failure_archive_n24 import prediction_sets


def sample(t,error,stratum):
    counts = []
    rng = np.random.default_rng(636)
    for k in np.unique(stratum[t]):
        group = t[stratum[t]==k]
        case,control = group[error[group]],group[~error[group]]
        if len(case):
            counts.append((case,np.ones(len(case))))
        if len(control):
            n = min(len(control),max(5,3*len(case)))
            ids = rng.choice(control,n,replace=False)
            counts.append((ids,np.full(n,len(control)/n)))
    ids,weight = [np.concatenate(x) for x in zip(*counts,strict=True)]
    np.testing.assert_allclose(weight.sum(),len(t))
    return ids,weight


def fit(x,error,t,q,stratum,matched):
    if matched:
        ids,w = sample(t,error,stratum)
    else:
        ids,w = t,np.ones(len(t))
    scale = StandardScaler().fit(x[t])
    a,b = [scale.transform(x[s]).clip(-6,6) for s in (ids,q)]
    classifier = LogisticRegression(C=.06,max_iter=400).fit(a,error[ids],sample_weight=w)
    return classifier.predict_proba(b)[:,1]


def run():
    raw,y,g,fold = data()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    cols,panels = prediction_sets(raw,y,g,fold,rows,of)
    family = np.array([k.split('/')[0] for k in cols])
    personal = design(raw)
    report = {}
    repeat = {}
    for (tag,ix,f,base,p),(six,arch) in zip([('full',np.arange(len(y)),fold,ref,full[:,names.index('mitra96')]),
                             ('historical',rows,of,champ,old[:,0])],panels,strict=True):
        e = base!=y[ix]
        against = np.column_stack([six,arch])
        dissent = (against>=.5)!=base[:,None]
        strict = ~dissent.any(1)
        # Count model families rather than raw columns to avoid overweighting larger families.
        family_dissent = np.column_stack([dissent[:,:3].any(1),dissent[:,3:6].any(1),
                *[dissent[:,6:][:,family==k].any(1) for k in np.unique(family)]])
        nf = family_dissent.sum(1)
        conf = np.where(base,p,1-p)
        bands = np.digitize(conf,[.6,.7,.8,.9,.97])
        stratum = bands+6*base.astype(int)
        summaries = [nf,conf,base]
        for values in (six[:,:3],six[:,3:],*[arch[:,family==k] for k in np.unique(family)]):
            opposition = np.where(base[:,None],1-values,values)
            z = logit(opposition.clip(.003,.997))
            summaries.extend([z.mean(1),z.max(1)-z.min(1)])
        x = np.column_stack([*summaries,personal[ix]])
        scores = {'baseline':1-conf}
        for level in np.unique(f):
            pool = np.flatnonzero(f==level)
            for a,b in GroupKFold(4,shuffle=True,random_state=136).split(pool,groups=g[ix][pool]):
                t,q = pool[a],pool[b]
                assert not np.isin(g[ix][t],g[ix][q]).any()
                for name,scope,matched in [('all',t,False),('strict',t[strict[t]],False),
                         ('near_consensus',t[nf[t]<=2],False),('near_matched',t[nf[t]<=2],True)]:
                    vals = scores.setdefault(name,np.full(len(ix),np.nan))
                    vals[q] = fit(x,e,scope,q,stratum,matched)
        info = {}
        for name,risk in scores.items():
            assert np.isfinite(risk).all()
            risk_scope = strict if name!='all' else np.ones(len(ix),bool)
            actions = {'eligible_majority':strict & (risk>.5),'scope_majority':risk_scope & (risk>.5)}
            for rate in (.05,.10):
                chosen = np.zeros(len(ix),bool)
                for level in np.unique(f):
                    ids = np.flatnonzero((f==level) & strict)
                    chosen[ids[np.argsort(-risk[ids],kind='stable')[:int(np.ceil(rate*len(ids)))]]] = True
                actions[str(rate)] = chosen
            info[name] = dict(auc=float(roc_auc_score(e[strict],risk[strict])),
                              average_precision=float(average_precision_score(e[strict],risk[strict])),actions={})
            for k,m in actions.items():
                info[name]['actions'][k] = dict(fixed=int((m&e).sum()),harmed=int((m&~e).sum()),strict_fixed=int((m&e&strict).sum()))
                key = name+'/'+k
                if tag=='full':
                    repeat[key]=[(m&e)[rows],(m&~e)[rows]]
                else:
                    info[name]['actions'][k].update(repeated_fixed=int((repeat[key][0]&m&e).sum()),
                                                   repeated_harmed=int((repeat[key][1]&m&~e).sum()))
        desc = {}
        for sign in (False,True):
            buckets = {}
            for b in range(6):
                s = strict & (base==sign) & (bands==b)
                buckets[str(b)] = [int(s.sum()),int((s&e).sum())]
            desc[str(sign)] = buckets
        report[tag] = dict(strict_n=int(strict.sum()),strict_errors=int((strict&e).sum()),
                           layout=desc,methods=info)
        print(tag,strict.sum(),(strict&e).sum(),[(name,round(r['auc'],3),r['actions']['0.1']['fixed'],r['actions']['eligible_majority']) for name,r in info.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/consensus_case_control.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
