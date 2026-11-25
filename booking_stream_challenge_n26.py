"""Challenge the fare-based release using registration and the order of independent bookings."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from exaone_validation_data import compact_features
from audit_fare_routes_n26 import DECK, held_fare
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker
from solo_selection_likelihood_n26 import fit_model
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits

OUT=ROOT/'artifacts/nov26_booking_stream_challenge'
REPORT=ROOT/'reports/nov23_matched_learning_curve/booking_stream_challenge.json'


def stream(frame):
    party=frame.PassengerId.str[:4]
    reg=compact_features(frame).drop(columns=FEES)
    reg['party_size']=party.map(party.value_counts()).to_numpy()
    reg['order_number']=party.astype(int).to_numpy()
    lane=reg.Deck+'_'+reg.Side
    item=pd.get_dummies(pd.DataFrame(dict(lane=lane,origin=reg.HomePlanet,destination=reg.Destination)),dtype=float)
    # Each booking contributes its distribution across locations once, regardless of family size.
    features=item.groupby(party,sort=True).mean()
    ix=features.index.get_indexer(party)
    roster=features.to_numpy()
    n=len(features)
    cumulative=np.concatenate([np.zeros((1,roster.shape[1])),np.cumsum(roster,axis=0)])
    order=np.arange(n)
    summaries={}
    controls={}
    rng=np.random.default_rng(7620)
    for radius in [8,40]:
        left,right=(order-radius).clip(0,n),(order+radius+1).clip(0,n)
        before=(cumulative[order]-cumulative[left])/np.maximum(1,order-left)[:,None]
        after=(cumulative[right]-cumulative[order+1])/np.maximum(1,right-order-1)[:,None]
        block=np.column_stack([before+after,after-before])/2
        permutation=order.copy()
        for idx in np.array_split(order,max(1,n//160)):
            offset=rng.integers(radius+1,len(idx)-radius)
            permutation[idx]=np.roll(idx,offset)
        assert (np.abs(permutation-order)>radius).all()
        for j in range(block.shape[1]):
            name=f'{radius}_'+('mean_' if j<roster.shape[1] else 'change_')+str(features.columns[j%roster.shape[1]])
            summaries[name]=block[ix,j]
            controls[name]=block[permutation[ix],j]
    return reg, pd.DataFrame(summaries), pd.DataFrame(controls)


def train():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    reg,local,sham=stream(pd.concat([raw,test],ignore_index=True))
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    extended=pd.concat([reg,local],axis=1)
    control=pd.concat([reg,sham],axis=1)
    OUT.mkdir(exist_ok=True)
    for path in sorted(DECK.glob('tree_*.npz')):
        with np.load(path) as z:
            t,q=z['t'],z['q']
            np.testing.assert_array_equal(z['y'],y[t])
        assert not np.isin(g[t],g[q]).any()
        p={}
        for name,x,sel in [('register_all',reg,t),('register_paid',reg,t[paid[t]]),
                           ('sequence_paid',extended,t[paid[t]]),('shuffled_paid',control,t[paid[t]])]:
            p[name]=fit_model(x.iloc[sel],y[sel]).predict_proba(x.iloc[q])[:,1]
        np.savez_compressed(OUT/path.name,q=q,**p)


def score():
    raw,y,g,_=data()
    v=sources()[-1]
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
        agree=(z['label'][:len(y)]>=0)&(z['fine'][:len(y)].argmax(1)==z['label'][:len(y)])
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    pred={}
    for tag,panel in v.items():
        preds=held_fare(panel,tag,with_routes=False)
        for fold in np.unique(panel['fold']):
            m=panel['fold']==fold
            with np.load(OUT/f'tree_{tag}_{fold}.npz') as z:
                np.testing.assert_array_equal(z['q'],panel['ids'][m])
                for n in z.files:
                    if n!='q':
                        preds.setdefault(n,np.zeros(len(m)))[m]=z[n]
        pred[tag]=preds
    bags={}
    for key,src,dst,t,q in splits(v,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            review=dst['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
        cur=pred['full']
        parent=combine(review,cur['neural_deck_raw'][q],cur['neural_deck_fine'][q],eligible[b])>=.5
        def matrix(panel,ps,idx,extras):
            native=panel['off'][idx]*np.where(panel['anchor'][idx],-1,1)
            cols=[native,panel['common'][idx],native*panel['common'][idx]]
            for n in ['tree_deck_fine','neural_deck_fine',*extras]:
                score=logit(np.clip(ps[n][idx],.01,.99))
                cols.extend([score,score*panel['common'][idx]])
            return np.column_stack(cols)
        votes={}
        challengers=['register_all','register_paid','sequence_paid','shuffled_paid']
        for name in ['native',*challengers]:
            extras=[] if name=='native' else [name]
            xt=matrix(src,pred['historical'],t,extras)
            xq=matrix(dst,cur,q,extras)
            fit=LogisticRegression(C=.3,max_iter=400).fit(xt[paid[a]],y[a][paid[a]])
            votes[name]=fit.predict(xq)
        actions={}
        for name in challengers:
            actions[name+'_direct']=paid[b]&((cur[name][q]>=.5)!=parent)
            actions[name+'_calibrated']=paid[b]&(votes['native']==parent)&(votes[name]!=parent)
        mode='bookingmix' if key.startswith('bookingmix') else key
        bags.setdefault(mode,[]).append((b,parent!=y[b],dst['common'][q],actions))
    report={}
    for key,items in bags.items():
        ids,wrong,com=[np.concatenate([b[j] for b in items]) for j in range(3)]
        assert len(ids)==len(np.unique(ids))
        masks={'all':np.ones(len(ids),bool),'consensus':com,'paid_agree_consensus':com&eligible[ids]&agree[ids]}
        report[key]={n:{k:ticker(mask,wrong,np.concatenate([b[3][n] for b in items])) for k,mask in masks.items()} for n in items[0][3]}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['bookingmix'],flush=True)


if __name__=='__main__':
    train()
    score()
