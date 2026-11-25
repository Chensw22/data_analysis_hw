"""Measure which changes to the current release survive disjoint training bookings."""
import json
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from full_roster_booking_validation_n23 import ROOT,data
from price_implied_deck_n26 import build_features, FEES
from fare_route_inference_n26 import route_features
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker
from outside_panel_coverage_n26 import archives, projection

OUT=ROOT/'artifacts/nov26_disjoint_challengers'
NAMES=['registration','ordinary_bill','implied_deck','implied_route']


def train():
    raw,y,g,fold=data()
    deck=build_features(raw,g)
    x={'registration':deck['raw'].drop(columns=[*FEES,'budget']),
       'ordinary_bill':deck['raw'],'implied_deck':deck['fine'],
       'implied_route':route_features(raw,g)['exact']}
    OUT.mkdir(exist_ok=True)
    for k in np.unique(fold):
        q=np.flatnonzero(fold==k)
        t=np.flatnonzero(fold!=k)
        order=np.unique(g[t])
        rng=np.random.default_rng(5287+int(k))
        first=np.isin(g[t],rng.permutation(order)[::2])
        groups=[t[first],t[~first]]
        assert not np.isin(g[groups[0]],g[groups[1]]).any() and not np.isin(g[t],g[q]).any()
        path=OUT/f'{k}.npz'
        if not path.exists():
            result=np.empty((len(q),len(NAMES),2))
            for j,rows in enumerate(groups):
                for i,name in enumerate(NAMES):
                    result[:,i,j]=fit_model(x[name].iloc[rows],y[rows]).predict_proba(x[name].iloc[q])[:,1]
            with path.with_suffix('.tmp').open('wb') as h:
                np.savez_compressed(h,first=groups[0],second=groups[1],q=q,
                                    training_y=y[np.r_[groups[0],groups[1]]],pred=result)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for name,value in [('first',groups[0]),('second',groups[1]),('q',q),
                               ('training_y',y[np.r_[groups[0],groups[1]]])]:
                np.testing.assert_array_equal(z[name],value)
        print(int(k),flush=True)


def score():
    raw,y,g,fold,views,original=projection()
    archive,_=archives(raw,y,g,fold,with_limix=True)
    families=np.column_stack([
        np.mean([archive[key] for key in ('Mitra1_frozen','Mitra2_frozen','Mitra2_adapted')],axis=0),
        archive['TabDPT'],archive['LimiX2'],archive['EXAONE_grouped']])
    p=np.empty((len(y),len(NAMES),2))
    for k in np.unique(fold):
        with np.load(OUT/f'{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],np.flatnonzero(fold==k))
            p[z['q']]=z['pred']
    odds=logit(p.clip(.005,.995))
    average=odds.mean(2)
    spread=np.abs(odds[:,:,0]-odds[:,:,1])
    # Estimate whether the amount of variation across fits contains information about the outcome.
    xbase=average
    xextra=np.column_stack([average,spread,spread*np.sign(average)])
    report={}
    for tag in ('bookingmix','historical_to_new'):
        mask,parent=views[tag]
        ids=np.flatnonzero(mask)
        wrong=parent[ids]!=y[ids]
        hard=(original[ids]==parent[ids,None]).all(1)
        actions={}
        votes=(p[ids]>=.5)!=parent[ids,None,None]
        repeats=votes.all(2)
        for i,n in enumerate(NAMES):
            actions[n+'_either']=votes[:,i].any(1)
            actions[n+'_both']=repeats[:,i]
        actions['any_repeat']=repeats.any(1)
        actions['all_repeat']=repeats.all(1)
        actions['at_least_two_repeat']=repeats.sum(1)>=2
        actions['registration_and_a_bill']=repeats[:,0]&repeats[:,1:].any(1)
        outside_votes=((families[ids]>=.5)!=parent[ids,None]).sum(1)
        for label,inside in [('no_filter',np.ones(len(ids),bool)),
                             ('one_repeat',repeats.any(1)),('two_repeat',repeats.sum(1)>=2)]:
            for minimum in (2,3,4):
                actions[f'{label}_outside_{minimum}']=inside&(outside_votes>=minimum)
        cal=np.full((len(y),2),np.nan)
        for k in np.unique(fold):
            rows=np.flatnonzero(fold==k)
            if tag=='bookingmix':
                parts=[(rows[t],rows[q][mask[rows[q]]]) for t,q in GroupKFold(3,shuffle=True,random_state=6391).split(rows,groups=g[rows])]
            else:
                parts=[(rows[~np.isin(g[rows],g[ids])],rows[mask[rows]])]
            for t,q in parts:
                if not len(q):continue
                assert not np.isin(g[t],g[q]).any()
                for j,xx in enumerate([xbase,xextra]):
                    model=LogisticRegression(C=.25,max_iter=500).fit(xx[t],y[t])
                    cal[q,j]=model.predict_proba(xx[q])[:,1]
        assert np.isfinite(cal[ids]).all()
        actions['mean_calibrated']=(cal[ids,0]>=.5)!=parent[ids]
        actions['spread_calibrated']=(cal[ids,1]>=.5)!=parent[ids]
        actions['spread_induced']=actions['spread_calibrated']&~actions['mean_calibrated']
        actions['spread_and_repeat']=actions['spread_induced']&actions['at_least_two_repeat']
        r={n:{s:ticker(m,wrong,a) for s,m in [('all',np.ones(len(ids),bool)),('all99',hard),('contested',~hard)]}
           for n,a in actions.items()}
        for name,action in actions.items():
            r[name]['net_by_fold']=[int(np.where(wrong,1,-1)[action&(fold[ids]==k)].sum()) for k in np.unique(fold)]
        report[tag]=r
        print(tag,[(n,v['all']['fixed'],v['all']['harmed']) for n,v in r.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/disjoint_challenger_repeats.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    train()
    score()
