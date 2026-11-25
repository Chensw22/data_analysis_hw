"""Replay consensus and calibrated decisions using registration models in their training domain."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from booking_stream_challenge_n26 import stream
from export_expert_vote_panel_n26 import OUT as TEST_PANEL, dig
from full_roster_booking_validation_n23 import ROOT, data
from held_expert_portfolios_n26 import NEW, outcome_scores, panel_predictions, sources, splits
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from solo_selection_likelihood_n26 import fit_model

REPORT=ROOT/'reports/nov23_matched_learning_curve'
CACHE=ROOT/'artifacts/nov26_domain_aware_consensus'
SCOPED=('register_paid','sequence_paid')


def features(scores,names,seq,paid):
    keep=~np.isin(names,SCOPED)
    assert scores.shape[1]==99 and keep.sum()==97
    idx=[names.index(n) for n in SCOPED]
    all_models=np.column_stack([scores[:,names.index('register_all')],seq])
    logits=logit(np.clip(scores,.002,.998))
    broad=logits[:,keep]
    # Common evidence is identical in all arms. The domain indicator is always visible.
    control=np.column_stack([np.quantile(broad,[0,.1,.5,.9,1],axis=1).T,
                             (broad>=0).mean(1),logits[:,names.index('register_all')],paid])
    raw=scores[:,idx]
    broad_vote=scores[:,keep]>=.5
    variants={}
    for tag,extra in [('original',raw),('abstain',np.where(paid[:,None],raw,.5)),
                      ('replace',np.where(paid[:,None],raw,all_models))]:
        low=np.minimum(broad_vote.min(1),(extra>=.5).min(1))
        high=np.maximum(broad_vote.max(1),(extra>=.5).max(1))
        if tag=='abstain':
            low[~paid]=broad_vote[~paid].min(1)
            high[~paid]=broad_vote[~paid].max(1)
        variants[tag]=(np.column_stack([control,logit(extra.clip(.002,.998))]),low,high)
    return variants


def counts(criterion,wrong):
    return {'rows':int(criterion.sum()),'fixed':int((criterion&wrong).sum()),'harmed':int((criterion&~wrong).sum())}


def run():
    raw,y,g,fold,views,official=projection()
    panel=sources()[-1]
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people=pd.concat([raw,test],ignore_index=True)
    paid=people[FEES].gt(0).any(axis=1).to_numpy()
    reg,local,_=stream(people)
    seq=pd.concat([reg,local],axis=1)
    names=[*panel['full']['names'].tolist(),*NEW]
    design={}
    CACHE.mkdir(exist_ok=True)
    for tag in ['full','historical','test']:
        if tag=='test':
            path=TEST_PANEL/'scores.npz'
            assert dig(path)==json.loads((REPORT/'competition_expert_panel.json').read_text())['sha256']
            with np.load(path) as z:
                np.testing.assert_array_equal(z['passenger'],test.PassengerId)
                np.testing.assert_array_equal(z['names'],names)
                scores,test_parent=z['scores'],z['prediction']
            ids=np.arange(len(y),len(people))
            parts=[(np.arange(len(y)),np.arange(len(test)))]
        else:
            v=panel[tag]
            ids=v['ids']
            scores=outcome_scores(v,panel_predictions(v,tag))
            if tag=='full':np.testing.assert_array_equal(scores>=.5,official)
            parts=[]
            for k in np.unique(v['fold']):
                q=np.flatnonzero(v['fold']==k)
                other=ids[v['fold']!=k]
                parts.append((other[~np.isin(g[other],g[ids[q]])],q))
        sequence=np.full(len(ids),np.nan)
        for k,(t,q) in enumerate(parts):
            rows=ids[q]
            assert not np.isin(people.PassengerId.str[:4].to_numpy()[rows],g[t]).any()
            path=CACHE/f'{tag}_{k}.npz'
            if not path.exists():
                p=fit_model(seq.iloc[t],y[t]).predict_proba(seq.iloc[rows])[:,1]
                with path.with_suffix('.tmp').open('wb') as fh:
                    np.savez_compressed(fh,t=t,q=rows,y=y[t],seq=p)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for name,val in [('t',t),('q',rows),('y',y[t])]:np.testing.assert_array_equal(z[name],val)
                sequence[q]=z['seq']
        assert np.isfinite(sequence).all()
        design[tag]=features(scores,names,sequence,paid[ids])
    methods=list(design['full'])
    predictions={key:{n:np.zeros(len(y),bool) for n in methods} for key in ['bookingmix','historical_to_new']}
    seen={key:np.zeros(len(y),bool) for key in predictions}
    for key,src,dst,t,q in splits(panel,g):
        mode='bookingmix' if key.startswith('bookingmix') else key
        if mode not in predictions:continue
        a,b=src['ids'][t],dst['ids'][q]
        assert not seen[mode][b].any() and not np.isin(g[a],g[b]).any()
        seen[mode][b]=True
        for n in methods:
            x=design['historical'][n][0][t]
            scaler=StandardScaler().fit(x)
            clf=LogisticRegression(C=.2,max_iter=600).fit(scaler.transform(x),y[a])
            predictions[mode][n][b]=clf.predict(scaler.transform(design['full'][n][0][q]))
    results={}
    for tag in ['bookingmix','historical_to_new','test']:
        if tag=='test':
            mask=np.ones(len(test),bool);parent=test_parent;blocks=design['test']
        else:
            mask,parent=views[tag];blocks=design['full']
            np.testing.assert_array_equal(mask,seen[tag])
        groups={n:mask&(parent==lo)&(parent==hi) for n,(_,lo,hi) in blocks.items()}
        r={}
        for n,hard in groups.items():
            cohorts={'all':mask,'positive_common':hard&parent,'negative_common':hard&~parent,
                     'added_common':hard&~groups['original'],'lost_common':groups['original']&~hard}
            if tag=='test':r[n]={key:int(m.sum()) for key,m in cohorts.items()};continue
            wrong=parent!=y
            flip=predictions[tag][n]!=parent
            raw_flip=predictions[tag]['original']!=parent
            r[n]={key:dict(rows=int(m.sum()),errors=int((m&wrong).sum()),edits=counts(m&flip,wrong)) for key,m in cohorts.items()}
            r[n]['new_edits']=counts(mask&flip&~raw_flip,wrong)
            r[n]['withdrawn_edits']=counts(mask&raw_flip&~flip,wrong)
            if n=='abstain':
                added=cohorts['added_common']
                fields={'deck':raw.Cabin.str[0], 'origin':raw.HomePlanet, 'sleep':raw.CryoSleep}
                r[n]['added_profile']={}
                for field,values in fields.items():
                    items=values.astype('string').fillna('?').to_numpy()
                    r[n]['added_profile'][field]={
                        str(value):{'rows':int((added&(items==value)).sum()),
                                    'errors':int((added&(items==value)&wrong).sum())}
                        for value in np.unique(items[added])}
        results[tag]=r
    (REPORT/'domain_aware_consensus.json').write_text(json.dumps(results,indent=2)+'\n')
    print(results,flush=True)


if __name__=='__main__':run()
