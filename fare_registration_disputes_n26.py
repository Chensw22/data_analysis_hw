"""Learn which side wins disagreements between registration-only experts and fare-based experts."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from audit_fare_routes_n26 import DECK, held_fare
from booking_stream_challenge_n26 import OUT as STREAM
from export_fare_deck_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from receipt_charge_swaps_n26 import OUT as SWAP
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits


def panel_predictions(v,tag):
    p=held_fare(v,tag)
    for fold in np.unique(v['fold']):
        m=v['fold']==fold
        for directory,names in [(STREAM,['register_all','register_paid','sequence_paid']),
                                (SWAP,['exact'])]:
            with np.load(directory/f'tree_{tag}_{fold}.npz') as z:
                np.testing.assert_array_equal(z['q'],v['ids'][m])
                for n in names:
                    p.setdefault('swap' if n=='exact' else n,np.zeros(len(m)))[m]=z[n]
    return p


def design(v,p,idx,challengers,x):
    prior=p['neural_deck_fine'][idx]
    judges=np.column_stack([p[n][idx] for n in challengers])
    direction=np.where(prior>=.5,-1,1)
    odds=lambda z:logit(np.clip(z,.01,.99))
    signed=direction[:,None]*odds(judges)
    flags=signed>0
    basic=np.column_stack([direction,-abs(odds(prior)),direction*odds(p['tree_deck_fine'][idx]),
                           direction*v['off'][idx]*np.where(v['anchor'][idx],-1,1),v['common'][idx],
                           flags.mean(1),signed.mean(1)])
    return np.column_stack([basic,signed,flags,signed*direction[:,None],x[v['ids'][idx]]]),flags.any(1)


def run():
    raw,y,g,_=data()
    v=sources()[-1]
    pred={tag:panel_predictions(panel,tag) for tag,panel in v.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
        agree=(z['label'][:len(y)]>=0)&(z['label'][:len(y)]==z['fine'][:len(y)].argmax(1))
    lane=raw.Cabin.str.split('/',expand=True)
    cats=pd.get_dummies(pd.DataFrame({'deck':lane[0],'side':lane[2],'origin':raw.HomePlanet,'dest':raw.Destination}),dtype=float)
    x=np.column_stack([cats,agree,raw.Age.fillna(28)/35])
    registry=['register_all','register_paid','sequence_paid']
    bills=['neural_route_exact','tree_route_exact','swap']
    bundles={}
    for key,src,dst,t,q in splits(v,g):
        if not (key.startswith('bookingmix') or key=='historical_to_new'):
            continue
        a,b=src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            review=dst['anchor'][q]^((z['identities_error']>.5)&(z['identities_outcome']>.5))
        old,cur=pred['historical'],pred['full']
        fare=combine(review,cur['neural_deck_raw'][q],cur['neural_deck_fine'][q],eligible[b])>=.5
        train_error=(old['neural_deck_fine'][t]>=.5)!=y[a]
        applicable=(cur['neural_deck_fine'][q]>=.5)==fare
        acts={}
        for n,judges in [('registry',registry),('receipts',bills),('together',registry+bills)]:
            xt,dt=design(src,old,t,judges,x)
            xq,dq=design(dst,cur,q,judges,x)
            fit_rows=dt&eligible[a]
            query=dq&eligible[b]&applicable
            model=LogisticRegression(C=.18,max_iter=600).fit(xt[fit_rows],train_error[fit_rows])
            acts[n]=query&(model.predict_proba(xq)[:,1]>.5)
            acts[n+'_opportunity']=query
        # A fixed majority within each input family avoids giving three registration
        # variants three times the influence of the independent receipt evidence.
        disagree=lambda cols:np.column_stack([((cur[n][q]>=.5)!=fare) for n in cols]).sum(1)
        reg_count,bill_count=disagree(registry),disagree(bills)
        acts['both_majorities']=eligible[b]&(reg_count>=2)&(bill_count>=2)
        acts['both_unanimous']=eligible[b]&(reg_count==3)&(bill_count==3)
        mode='bookingmix' if key.startswith('bookingmix') else key
        bundles.setdefault(mode,[]).append((b,fare!=y[b],dst['common'][q],acts))
    report={}
    for mode,items in bundles.items():
        ids,wrong,com=[np.concatenate([it[j] for it in items]) for j in range(3)]
        assert len(ids)==len(np.unique(ids))
        masks={'all':np.ones(len(ids),bool),'consensus':com,'paid_agree_consensus':com&eligible[ids]&agree[ids]}
        report[mode]={n:{key:ticker(m,wrong,np.concatenate([it[3][n] for it in items])) for key,m in masks.items()}
                      for n in items[0][3]}
        print(mode,report[mode]['together'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/fare_registration_disputes.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
