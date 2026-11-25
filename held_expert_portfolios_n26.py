"""Separate the achievable coverage of complementary expert votes from deployable expert selection."""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler

from audit_fare_routes_n26 import DECK
from export_fare_deck_n26 import combine as fare_model
from export_six_way_dispute_n26 import combine as six_model
from fare_registration_disputes_n26 import panel_predictions
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, ROOT, sources, splits

NEW=['neural_deck_raw','tree_deck_raw','neural_deck_fine','tree_deck_fine','neural_route_exact','tree_route_exact',
     'swap','register_all','register_paid','sequence_paid']
SIX=['register_all','register_paid','sequence_paid','tree_route_exact','swap','neural_route_exact']
REPORT=ROOT/'reports/nov23_matched_learning_curve/held_expert_portfolios.json'


def outcome_scores(v,p):
    direction=np.where(v['anchor'],-1,1)
    center=v['off']+v['basic'][:,3]
    old=expit((center[:,None]+v['relative'])*direction[:,None])
    # At exactly 0.5 the stored original vote disambiguates floating-point roundoff.
    mismatch=(old>=.5)!=v['votes']
    assert np.all(np.abs(old[mismatch]-.5)<1.e-5)
    old[mismatch]=np.where(v['votes'][mismatch],.5,np.nextafter(.5,0))
    return np.column_stack([old,*[p[n] for n in NEW]])


def greedy_pool(p,truth):
    vote=p>=.5
    ok=vote==truth[:,None]
    # Begin with the best independent score, then ask which new vote covers the most misses.
    best=int(ok.sum(0).argmax())
    chosen=[best]
    covered=ok[:,best].copy()
    for _ in range(7):
        gains=ok[~covered].sum(0).astype(float)
        gains[chosen]=-1
        if gains.max()<=0:
            break
        nxt=int(np.argmax(gains+ok.mean(0)*1.e-3))
        covered |= ok[:,nxt]
        chosen.append(nxt)
    return chosen


def physical(frame):
    cats=pd.get_dummies(frame[['Deck','Side','HomePlanet','Destination','CryoSleep','VIP']].astype(str),dtype=float)
    numbers=frame[['Age','CabinNumber','booking_size',*FEES]].copy()
    numbers[FEES]=np.log1p(numbers[FEES])
    return np.column_stack([cats,numbers.isna().astype(float),numbers.fillna(-1)])


def standardized(a,b):
    scaler=StandardScaler().fit(a)
    return scaler.transform(a).clip(-5,5),scaler.transform(b).clip(-5,5)


def distinct_neighbors(xt,xq,groups):
    distances=cdist(xq,xt,'sqeuclidean')
    nearest=np.empty((len(xq),24),int)
    for i,order in enumerate(np.argsort(distances,axis=1,kind='stable')):
        # The closest member represents the booking when estimating local competence.
        _,first=np.unique(groups[order],return_index=True)
        nearest[i]=order[np.sort(first)[:24]]
    return nearest


def selectors(pt,pq,yt,near):
    votes=pt>=.5
    correct=(votes==yt[:,None])
    local=correct[near]
    # Longest ordered prefix of unlike bookings on which an expert is correct.
    prefix=np.cumprod(local[:,:8],axis=1).sum(1)
    keep=prefix==prefix.max(1,keepdims=True)
    weights=(local.sum(1)+2*correct.mean(0)+1)
    direct=(pq*keep).sum(1)/keep.sum(1)
    winner=weights.argmax(1)
    return {'prefix_consensus':direct>=.5,'local_winner':pq[np.arange(len(pq)),winner]>=.5}


def replay(key,q,v,p,enabled):
    ids=v['ids'][q]
    with np.load(REVIEW/f'pred_{key}.npz') as z:
        np.testing.assert_array_equal(z['q'],ids)
        review=v['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
    fare=fare_model(review,p['neural_deck_raw'][q],p['neural_deck_fine'][q],enabled[ids])
    challengers=np.column_stack([p[n][q] for n in SIX])
    return six_model(fare,challengers,v['votes'][q,:18],enabled[ids])>=.5


def run():
    raw,y,g,panels=sources()
    p={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    scores={tag:outcome_scores(v,p[tag]) for tag,v in panels.items()}
    names=[*panels['full']['names'],*NEW]
    rawx=physical(raw)
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
        agrees=(z['label']>=0)&(z['fine'].argmax(1)==z['label'])
    batches={}
    for key,src,dst,t,q in splits(panels,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        pt,pq=scores['historical'][t],scores['full'][q]
        chosen=greedy_pool(pt,y[a])
        baseline=replay(key,q,dst,p['full'],enabled)
        vote=pq>=.5
        acts={'any_of_all':(vote!=baseline[:,None]).any(1),
              'any_of_selected':(vote[:,chosen]!=baseline[:,None]).any(1),
              'historical_best':vote[:,chosen[0]]!=baseline,
              'selected_mean':(pq[:,chosen].mean(1)>=.5)!=baseline}
        # Absolute confidence and disagreement describe different neighborhoods.
        ct,cq=standardized(logit(pt[:,chosen].clip(.004,.996)),logit(pq[:,chosen].clip(.004,.996)))
        rt,rq=standardized(rawx[a],rawx[b])
        params=[('registered',rt,rq),('scores',ct,cq),
                ('joint',np.column_stack([rt/np.sqrt(rt.shape[1]),ct/np.sqrt(ct.shape[1])]),
                         np.column_stack([rq/np.sqrt(rq.shape[1]),cq/np.sqrt(cq.shape[1])]))]
        for mode,xtrain,xquery in params:
            near=distinct_neighbors(xtrain,xquery,g[a])
            for n,d in selectors(pt[:,chosen],pq[:,chosen],y[a],near).items():
                acts[mode+'_'+n]=d!=baseline
        tag='bookingmix' if key.startswith('bookingmix') else key
        batches.setdefault(tag,[]).append((b,baseline!=y[b],dst['common'][q],acts,[str(names[i]) for i in chosen]))
    report={}
    for tag,entries in batches.items():
        ids,wrong,common=[np.concatenate([e[i] for e in entries]) for i in range(3)]
        assert len(np.unique(ids))==len(ids)
        scopes={'all':np.ones(len(ids),bool),'previous_unanimity':common,'other':~common,
                'previous_unanimity_paid_deck_agrees':common&enabled[ids]&agrees[ids]}
        out={n:{k:ticker(m,wrong,np.concatenate([e[3][n] for e in entries])) for k,m in scopes.items()} for n in entries[0][3]}
        out['selected_on_training_orders']=[e[4] for e in entries]
        report[tag]=out
        print(tag,[(k,v['all']['fixed'],v['all']['harmed']) for k,v in out.items() if isinstance(v,dict)],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
