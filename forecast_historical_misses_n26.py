"""Predict historical shared-error risk without using the evaluated booking's labels."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exhaustive_family_quorums_n26 import summary
from latent_passenger_difficulty_n26 import OUT, ROOT
from nori_fare_party_n26 import FEES


def run():
    with np.load(OUT/'panel.npz') as z:
        y,f,anchor,known,common,g,regime,score,ids,fresh=[z[k] for k in
            ('y','fold','anchor','known','common','groups','regime','scores','passenger','fresh')]
    n=len(y)
    anchor,g,regime,score=anchor[:n],g[:n],regime[:n],score[:n]
    assert np.all((regime%2==0)==pd.Series(g).groupby(g).transform('size').eq(1))
    with np.load(ROOT/'artifacts/nov26_nori_adjacent_manifest/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        np.testing.assert_array_equal(z['reference161'],anchor)
        real,original,placebo=[z[k]>=.5 for k in ('real','raw','permuted')]
        observed=z['observed']
        unique=(real!=anchor)&(original==anchor)&(placebo==anchor)&observed
    catalog=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',usecols=['row_id','PassengerId','y','all_recipes_still_wrong'])
    idx=catalog.row_id.to_numpy()
    np.testing.assert_array_equal(ids[idx],catalog.PassengerId)
    np.testing.assert_array_equal(y[idx],catalog.y)
    historical=np.full(n,-1)
    historical[idx]=catalog.all_recipes_still_wrong.to_numpy(int)
    assert np.count_nonzero(historical==1)==546
    raw=pd.read_csv(ROOT/'data/raw/train.csv')
    np.testing.assert_array_equal(raw.PassengerId,ids[:n])
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    wrong=anchor!=y
    # No validation truths, old miss flags, or original baseline directions enter these features.
    margin=np.abs(score-.5)
    design=np.column_stack([score.mean(axis=1),margin.mean(axis=1),score.std(axis=1),
        score[:,-3:],margin[:,-3:],np.eye(6)[regime],raw.Age.fillna(-1),np.log1p(raw[FEES].fillna(-1).clip(lower=0)),
        raw[FEES].isna(),pd.get_dummies(raw.HomePlanet,dtype=float),pd.get_dummies(raw.Cabin.str[0],dtype=float)])
    report={}
    for ordered in [False,True]:
        risk={k:np.full(n,np.nan) for k in ['historical','current']}
        selected={k:np.zeros(n,bool) for k in risk}
        for outer in np.unique(f):
            rows=np.flatnonzero(f==outer)
            if ordered:
                blocks=np.array_split(np.unique(g[rows]),4)
                splits=[(rows[~np.isin(g[rows],b)],rows[np.isin(g[rows],b)]) for b in blocks]
            else:
                splits=[(rows[a],rows[b]) for a,b in GroupKFold(4,shuffle=True,random_state=703+int(outer)).split(rows,groups=g[rows])]
            for t,q in splits:
                fit=t[(historical[t]>=0)&known[t]]
                assert not np.isin(g[fit],g[q]).any()
                for name,labels in [('historical',historical),('current',wrong)]:
                    model=make_pipeline(StandardScaler(),LogisticRegression(C=.08,max_iter=400))
                    model.fit(design[fit],labels[fit])
                    risk[name][q]=model.predict_proba(design[q])[:,1]
                    cutoff=np.quantile(model.predict_proba(design[fit])[:,1],.75)
                    selected[name][q]=risk[name][q]>=cutoff
        assert all(np.isfinite(a).all() for a in risk.values())
        act=dict(all_unique=unique,solo_no_positive=unique&~paid&(regime%2==0),
                 historical_risk=unique&selected['historical'], current_risk=unique&selected['current'])
        views=dict(all=known,common=common,unseen_orders=known&fresh, archived_misses=known&(historical==1))
        report['ordered' if ordered else 'shuffled']={k:{v:summary(a,wrong,m,f) for v,m in views.items()} for k,a in act.items()}
    path=ROOT/'reports/nov23_matched_learning_curve/forecast_historical_misses.json'
    for k,r in report.items():
        print(k,[(v,a['all']['fixed'],a['all']['harmed'],a['unseen_orders']['net']) for v,a in r.items()])
    branches=pd.read_csv(ROOT/'artifacts/nov26_clean_fare_unknown_branches/paired_rows.csv')
    np.testing.assert_array_equal(branches.PassengerId,ids[:n][~known])
    deltas=[]
    for key in ('161_low','161_high'):
        base=anchor.copy()
        base[~known]=branches[key].to_numpy(bool)
        change=~paid&(regime%2==0)&observed&(real!=base)&(original==base)&(placebo==base)
        deltas.append(np.where(change,np.where(base!=y,1,-1),0))
    assert deltas[0][known].sum()==6 and np.array_equal(deltas[0][known],deltas[1][known])
    report['solo_no_positive_full_net_bounds']=[int(np.min(deltas,axis=0).sum()),int(np.max(deltas,axis=0).sum())]
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
