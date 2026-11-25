"""Use historical orders in which experts from both sides of a current dispute made matching calls."""
import json
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from held_expert_portfolios_n26 import DECK, ROOT, outcome_scores, panel_predictions, replay, sources, splits, ticker


def paired_local(train,query,pt,pq,y,group):
    # Discount frequently repeated expert outputs using correlations of their deviations from the median.
    residual=train-np.median(train,axis=1,keepdims=True)
    corr=np.nan_to_num(np.corrcoef(residual.T),nan=0.)
    weights=1/np.maximum(1,(corr>.9).sum(1))
    weights/=weights.sum()
    tv=(train>=.5).astype(float)
    qv=(query>=.5).astype(float)
    counts=np.unique(group,return_inverse=True,return_counts=True)
    multiplicity=counts[2][counts[1]]
    result={k:np.zeros(len(query)) for k in ('baseline','pairs','identity_control','single_camp')}
    rng=np.random.default_rng(4471)
    perm=rng.permutation(query.shape[1])
    for start in range(0,len(query),100):
        stop=min(start+100,len(query))
        ix=np.arange(start,stop)
        neighborhood=np.exp(-.5*((pq[ix,None]-pt[None,:])/.8)**2)/multiplicity
        # Estimate a local log-odds adjustment to the ordinary native probability.
        same_baseline=expit(pt)
        prior=expit(pq[ix])
        def adjust(w):
            weight=w.sum(1)
            observed=(w@y+8*prior)/(weight+8)
            expected=(w@same_baseline+8*prior)/(weight+8)
            return expit(pq[ix]+logit(observed.clip(.001,.999))-logit(expected.clip(.001,.999)))
        result['baseline'][ix]=adjust(neighborhood)
        for label,select in [('pairs',qv[ix]),('identity_control',qv[ix][:,perm])]:
            pos=select*weights
            neg=(1-select)*weights
            a=(pos@tv.T)/np.maximum(pos.sum(1,keepdims=True),1.e-12)
            b=(neg@(1-tv).T)/np.maximum(neg.sum(1,keepdims=True),1.e-12)
            # Four draws from each side reward orders on which much of both camps agreed.
            # Unanimous queries have no opposing pair and retain their calibrated baseline.
            both=(pos.sum(1)>0)&(neg.sum(1)>0)
            score=adjust(neighborhood*np.power(a*b,4))
            result[label][ix]=np.where(both,score,result['baseline'][ix])
            if label=='pairs':
                result['single_camp'][ix]=np.where(both,adjust(neighborhood*np.power((a+b)/2,8)),result['baseline'][ix])
    return result


def run():
    _,y,g,panels=sources()
    p={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    scores={tag:outcome_scores(v,p[tag]) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
        agrees=(z['label']>=0)&(z['fine'].argmax(1)==z['label'])
    collected={}
    native=lambda v,idx:v['off'][idx]*np.where(v['anchor'][idx],-1,1)
    for key,src,dst,t,q in splits(panels,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        parent=replay(key,q,dst,p['full'],enabled)
        forecasts=paired_local(scores['historical'][t],scores['full'][q],native(src,t),native(dst,q),y[a],g[a])
        tag='bookingmix' if key.startswith('bookingmix') else key
        collected.setdefault(tag,[]).append((b,parent,dst['common'][q],forecasts))
    report={}
    for tag,items in collected.items():
        ids,parent,common=[np.concatenate([v[i] for v in items]) for i in range(3)]
        assert len(np.unique(ids))==len(ids)
        pred={k:np.concatenate([v[3][k] for v in items]) for k in items[0][3]}
        actions={k:(v>=.5)!=parent for k,v in pred.items()}
        actions.update(pair_induced=actions['pairs']&~actions['baseline'],
                       pair_induced_over_single=actions['pairs']&~actions['single_camp']&~actions['baseline'])
        masks={'all':np.ones(len(ids),bool),'previous_unanimity':common,
               'paid_agree_unanimity':common&enabled[ids]&agrees[ids]}
        r={k:{name:ticker(m,parent!=y[ids],a) for name,m in masks.items()} for k,a in actions.items()}
        r['log_loss']={k:float(-np.log(np.where(y[ids],v,1-v)).mean()) for k,v in pred.items()}
        print(tag,r['pair_induced'],r['log_loss'],flush=True)
        report[tag]=r
    (ROOT/'reports/nov23_matched_learning_curve/dissenting_pairs_transfer.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
