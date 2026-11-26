"""Embed the manifest distribution of independent nearby bookings for each passenger."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from matched_state_context_n26 import ROOT, inputs
from nori_fare_party_n26 import FEES
from exhaustive_family_quorums_n26 import summary


def design(frame, groups):
    cabin = frame.Cabin.str.split('/',expand=True)
    cats = pd.DataFrame(dict(planet=frame.HomePlanet,dest=frame.Destination,vip=frame.VIP,
                             sleep=frame.CryoSleep,deck=cabin[0],side=cabin[2])).astype('string').fillna('?')
    bills = np.log1p(frame[FEES]).to_numpy(float)
    size = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
    pos = pd.to_numeric(cabin[1]).to_numpy()
    own = np.column_stack([pd.get_dummies(cats).to_numpy(float),frame.Age.to_numpy()/25,
                           np.log1p(size),bills/4,np.log1p(frame[FEES].sum(axis=1,min_count=1))/4])
    # The random features depend only on field values and fixed random numbers.
    # Their mean approximates a Gaussian-kernel embedding of each comparison set.
    rng = np.random.default_rng(2641)
    finite = np.column_stack([np.nan_to_num(own),np.isnan(own).astype(float)])
    basis = rng.normal(0,.7,(finite.shape[1],48))
    features = np.cos(finite@basis + rng.uniform(0,2*np.pi,48))
    nearby = np.full_like(features,np.nan)
    sham = nearby.copy()
    distance = np.full(len(own),np.nan)
    # Collapse occupants of the same booking at each cabin into a single mark.
    # For both libraries select one representative per distinct other order.
    for _, lane in frame.loc[cabin.notna().all(axis=1)].groupby([cabin[0],cabin[2]]):
        ix = lane.index.to_numpy()
        order = np.argsort(pos[ix],kind='stable')
        ix = ix[order]
        seats = pd.DataFrame(features[ix]).assign(group=groups[ix],pos=pos[ix]).groupby(['pos','group'],sort=True).mean()
        mp = seats.index.get_level_values('pos').to_numpy()
        mg = seats.index.get_level_values('group').to_numpy()
        marks = seats.to_numpy()
        # Circularly shift addresses within a lane while preserving marks and their correlations.
        for i in ix:
            for real, center in ((True,pos[i]),(False,mp[(np.searchsorted(mp,pos[i])+len(mp)//2)%len(mp)])):
                loc = np.searchsorted(mp,center)
                cand = np.arange(max(0,loc-60),min(len(mp),loc+61))
                ranked = cand[np.argsort(np.abs(mp[cand]-center),kind='stable')]
                ranked = ranked[mg[ranked]!=groups[i]]
                _, first = np.unique(mg[ranked],return_index=True)
                nearest = ranked[np.sort(first)[:12]]
                if not len(nearest):
                    continue
                (nearby if real else sham)[i] = marks[nearest].mean(0)
                if real:
                    distance[i] = np.abs(mp[nearest]-center).mean()
    return np.column_stack([own,pos,groups.astype(float),features,distance]),nearby,sham


def run():
    frame,y,g,f,masks = inputs()
    n = len(y)
    assert 'Name' not in frame.columns
    own,nearby,sham = design(frame,g)
    result = {}
    path = ROOT/'artifacts/nov26_cabin_manifest_mean_embeddings'
    path.mkdir(exist_ok=True)
    with np.load(ROOT/'artifacts/nov26_retained_order_mechanisms/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId[:n])
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_array_equal(z['y'],y)
        base = z['reference161']
        blind = z['eligible'] & ((z['original_panel']>=.5)==base[:,None]).all(1) & ((z['pairs']>=.5)==base[:,None,None]).all(axis=(1,2))
        known,fresh = z['known'],z['fresh']
    assert (blind & (base!=y)).sum()==266
    out = {}
    for name, x in [('own',own),('real',np.column_stack([own,nearby])),('shifted',np.column_stack([own,sham]))]:
        p = np.empty(len(frame))
        for k in range(9):
            t = np.flatnonzero(f!=k) if k<8 else np.arange(n)
            q = np.flatnonzero(f==k) if k<8 else np.arange(n,len(g))
            assert not np.isin(g[t],g[q]).any()
            model = HistGradientBoostingClassifier(max_iter=155,max_leaf_nodes=15,min_samples_leaf=24,
                         learning_rate=.062,l2_regularization=4,early_stopping=False,random_state=2614)
            model.fit(x[t],y[t])
            p[q] = model.predict_proba(x[q])[:,1]
        assert np.isfinite(p).all()
        out[name] = p
        print(name,float(((p[:n]>=.5)==y).mean()),flush=True)
    with np.load(ROOT/'artifacts/nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
        anchors = z['anchor']
        np.testing.assert_array_equal(anchors[:n],base)
    scope = dict(all=known,blind31=blind,fresh=known&fresh,awake=blind&masks['awake_paid'][:n],sleep=blind&masks['sleep_unpaid'][:n])
    a = {k:(v>=.5)!=anchors for k,v in out.items()}
    edits = dict(**a,real_only=a['real']&~a['own']&~a['shifted'])
    for k,v in edits.items():
        result[k]=dict(scopes={s:summary(v[:n],base!=y,m,f) for s,m in scope.items()},test_edits=int(v[n:].sum()))
    report=dict(accuracy={k:float(((p[:n]>=.5)==y).mean()) for k,p in out.items()},against_161=result)
    np.savez_compressed(path/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**out)
    (ROOT/'reports/nov23_matched_learning_curve/cabin_manifest_mean_embeddings.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
