"""Test whether service-specific integer price ratios reveal replicable fare levels beyond the bill total."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from exaone_validation_data import compact_features
from exhaustive_family_quorums_n26 import summary
from full_roster_booking_validation_n23 import ROOT, data
from latent_passenger_difficulty_n26 import OUT as PANEL
from price_implied_deck_n26 import DECKS, FEES
from solo_selection_likelihood_n26 import fit_model

OUT = ROOT/'artifacts/nov26_weighted_bill_ridges'
REPORT = ROOT/'reports/nov23_matched_learning_curve/weighted_bill_ridges.json'


def projections():
    yield 'original_total', np.ones(5, int)
    for j, name in enumerate(FEES):
        for label, other, chosen in [('double', 1, 2), ('half', 2, 1), ('exclude', 1, 0)]:
            weights = np.full(5, other)
            weights[j] = chosen
            yield label+'_'+name, weights


def deck_posterior(cost, bank, ask, deck, mean_weight, displaced):
    """Average counts per integer around the query under narrow and broad windows."""
    scaled = lambda width: max(1, int(round(width*mean_weight)))
    fine, coarse, band = map(scaled, [3, 80, 128])
    original = cost[ask]
    center = (original//band)*band + (original+scaled(43))%band if displaced else original
    density = []
    for j in range(len(DECKS)):
        grid = np.sort(cost[bank[deck[bank]==j]])
        def counts(query, radius):
            return np.searchsorted(grid,query+radius,side='right')-np.searchsorted(grid,query-radius)
        # Broad component is identical for the actual and displaced narrow peak.
        narrow = counts(center, fine)/(2*fine+1)
        wide = counts(original, coarse)/(2*coarse+1)
        density.append(.75*narrow+.25*wide+.00002*(len(grid)+1))
    d = np.column_stack(density)
    return d/d.sum(axis=1,keepdims=True)


def make_features(frame):
    fee = frame[FEES]
    bills = fee.fillna(0).to_numpy(int)
    parts = 2*bills+(bills>0)
    complete = fee.notna().all(axis=1).to_numpy()
    paid = (parts>0).any(axis=1)
    deck = pd.Index(DECKS).get_indexer(frame.Cabin.str[0])
    planet = frame.HomePlanet.fillna('?').to_numpy()
    ids = frame.PassengerId.str.split('_').str[0].to_numpy()
    splits = list(GroupKFold(5,shuffle=True,random_state=9664).split(frame,groups=ids))
    features = {}
    diagnostics = {}
    for name, weights in projections():
        cost = parts@weights
        eligible = complete & paid & (cost>0)
        # Every modified projection must have at least two nonzero charges on which its weights differ.
        if name!='original_total':
            different = np.zeros(len(frame),bool)
            for j in range(5):
                for k in range(j):
                    if weights[k]!=weights[j]:
                        different |= (parts[:,j]>0)&(parts[:,k]>0)
            eligible &= different
        prediction = np.full((2,len(frame),len(DECKS)),np.nan)
        for t,q in splits:
            assert not np.isin(ids[t],ids[q]).any()
            t = t[eligible[t]&(deck[t]>=0)]
            q = q[eligible[q]]
            for origin in np.unique(planet[q]):
                bank = t[planet[t]==origin]
                if len(bank)<30:
                    bank=t
                ask=q[planet[q]==origin]
                for s, displaced in enumerate([False,True]):
                    prediction[s,ask]=deck_posterior(cost,bank,ask,deck,weights.mean(),displaced)
        assert np.isfinite(prediction[:,eligible]).all()
        rows=np.flatnonzero(eligible&(deck>=0))
        reg=prediction[:,rows,deck[rows]].clip(1.e-12)
        diagnostics[name]=dict(rows=len(rows), accuracy=(prediction[:,rows].argmax(axis=2)==deck[rows]).mean(axis=1).tolist(),
                               logloss=(-np.log(reg)).mean(axis=1).tolist())
        registered = np.full((2,len(frame)),np.nan)
        registered[:,rows]=np.log(reg)
        features[name]=(prediction,registered)
    reference, ref_reg = features.pop('original_total')
    x = compact_features(frame)
    for j,name in enumerate(DECKS):
        x['total_deck_'+name] = reference[0,:,j]
    x['total_registered'] = ref_reg[0]
    result = {'total':x}
    for s, mode in enumerate(['weighted','displaced']):
        aux={}
        for name,(p,registered) in features.items():
            aux[name+'_log_registered']=registered[s]
            aux[name+'_peak']=np.max(p[s],axis=1)
        result[mode]=pd.concat([x,pd.DataFrame(aux)],axis=1)
    return result,diagnostics


def run():
    train,y,g,fold=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=train.columns.tolist())
    frame=pd.concat([train,test],ignore_index=True)
    design,diagnostics=make_features(frame)
    n=len(y)
    fit_test=np.arange(n,len(frame))
    with np.load(PANEL/'panel.npz') as z:
        for k,v in [('y',y),('fold',fold),('passenger',frame.PassengerId),('groups',frame.PassengerId.str.split('_').str[0])]:
            np.testing.assert_array_equal(z[k],v)
        anchor,known,common,fresh=[z[k] for k in ('anchor','known','common','fresh')]
    p={k:np.full(len(frame),np.nan) for k in design}
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        for name,x in design.items():
            p[name][q]=fit_model(x.iloc[t],y[t]).predict_proba(x.iloc[q])[:,1]
    for name,x in design.items():
        p[name][fit_test]=fit_model(x.iloc[:n],y).predict_proba(x.iloc[fit_test])[:,1]
        assert np.isfinite(p[name]).all()
    bill=frame[FEES]
    multi=(bill.gt(0).sum(axis=1)>=2).to_numpy()&bill.notna().all(axis=1).to_numpy()
    pos=bill.gt(0).any(axis=1).to_numpy()
    awake=frame.CryoSleep.eq(False).to_numpy()
    candidates={k:(v>=.5)!=anchor for k,v in p.items()}
    unique=multi&candidates['weighted']&~candidates['total']&~candidates['displaced']
    actions=dict(**candidates, weighted_only=unique, awake_negative_only=unique&awake&~anchor)
    masks=dict(all=known, common=common, fresh=known&fresh,
               common_awake_negative=common&awake[:n]&pos[:n]&~anchor[:n],
               eligible_awake_negative=common&awake[:n]&multi[:n]&~anchor[:n], multiple_charges=known&multi[:n])
    metrics={name:{s:summary(change[:n],anchor[:n]!=y,m,fold) for s,m in masks.items()} for name,change in actions.items()}
    accuracy={name:float(((v[:n]>=.5)==y).mean()) for name,v in p.items()}
    for k in ['total','displaced']:
        metrics['weighted_vs_'+k]={s:summary((p['weighted'][:n]>=.5)!=(p[k][:n]>=.5),(p[k][:n]>=.5)!=y,m,fold) for s,m in masks.items()}
    result=dict(deck_inference=diagnostics,accuracy=accuracy,relative_to_161=metrics,
                test_changes={k:int(v[n:].sum()) for k,v in actions.items()})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',passenger=frame.PassengerId.to_numpy(str),fold=fold,y=y,anchor=anchor,**p)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print('accuracy',accuracy)
    print([(k,v['all']['fixed'],v['all']['harmed'],v['common_awake_negative']['fixed'],v['common_awake_negative']['harmed']) for k,v in metrics.items()])


if __name__=='__main__':
    run()
