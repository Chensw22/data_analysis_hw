"""Infer deck from the sharp peaks in integer spending, then test the mismatch with the registered deck."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data

FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
DECKS=list('ABCDEFGT')
OUT=ROOT/'artifacts/nov26_price_implied_deck'
RESULT=ROOT/'reports/nov23_matched_learning_curve/price_implied_deck.json'


def prepare():
    raw,y,g,_=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people=pd.concat([raw,test],ignore_index=True)
    group=people.PassengerId.str.split('_').str[0].to_numpy()
    origin=people.HomePlanet.fillna('?').to_numpy()
    deck=people.Cabin.str.split('/').str[0].fillna('?').to_numpy()
    fees=people[FEES]
    count=fees.gt(0).sum(axis=1).to_numpy()
    amount=fees.fillna(0).sum(axis=1).to_numpy(int)
    eligible=fees.notna().all(axis=1).to_numpy()&(count>0)
    cost=2*amount+count
    limit=int(cost.max()+2*64+8)
    results={n:np.full((len(people),len(DECKS)),1/len(DECKS)) for n in ('coarse','fine','displaced')}
    fold=np.empty(len(people),int)
    for k,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=9664).split(people,groups=group)):
        assert not np.isin(group[t],group[q]).any()
        fold[q]=k
        support=t[eligible[t]&(deck[t]!='?')]
        for p in np.unique(origin[q]):
            bank=support[origin[support]==p]
            if len(bank)<25:bank=support
            ask=q[(origin[q]==p)&eligible[q]]
            if not len(ask):continue
            coarse=[];fine=[];control=[]
            bin_start=128*(amount[ask]//64)+count[ask]
            shifted=bin_start+2*((amount[ask]+23)%64)
            for d in DECKS:
                members=bank[deck[bank]==d]
                h=np.bincount(cost[members],minlength=limit).astype(float)
                wide=gaussian_filter1d(h,42,mode='constant')+.00002*(1+len(members))
                sharp=gaussian_filter1d(h,2.4,mode='constant')
                smooth=.78*sharp+.22*wide
                coarse.append(wide[cost[ask]])
                fine.append(smooth[cost[ask]])
                # Retain the coarse amount, while breaking its alignment with narrow peaks.
                control.append(wide[cost[ask]]*smooth[shifted]/wide[shifted])
            for n,probs in [('coarse',coarse),('fine',fine),('displaced',control)]:
                mat=np.stack(probs,axis=1)
                results[n][ask]=mat/mat.sum(1,keepdims=True)
    label=pd.Index(DECKS).get_indexer(deck)
    known=eligible&(label>=0)
    report={}
    for n,p in results.items():
        np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
        def measures(ids):
            return dict(rows=len(ids),accuracy=float(np.mean(p[ids].argmax(1)==label[ids])),
                        nll=float(-np.log(p[ids,label[ids]].clip(1e-9)).mean()))
        report[n]={s:measures(np.flatnonzero(known&m)) for s,m in [
            ('all',np.ones(len(known),bool)),('training',np.arange(len(known))<len(y)),
            ('competition',np.arange(len(known))>=len(y)),*[(p,origin==p) for p in ('Earth','Mars','Europa')]]}
        report[n]['fold_accuracy']=[measures(np.flatnonzero(known&(fold==k)))['accuracy'] for k in range(5)]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'deck_estimates.npz',passenger=people.PassengerId.to_numpy(str),label=label,
                        eligible=eligible,cost=cost,**results)
    (OUT/'deck_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print({n:r['all'] for n,r in report.items()},flush=True)


def restore_features():
    raw,y,g,f=data()
    return y,g,f,build_features(raw,g)


def build_features(raw,g):
    x=compact_features(raw)
    x['booking_size']=raw.PassengerId.groupby(g).transform('size')
    with np.load(OUT/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(raw)],raw.PassengerId)
        eligible=z['eligible'][:len(raw)]
        known=z['label'][:len(raw)]>=0
        label=np.maximum(z['label'][:len(raw)],0)
        for_direct=x.assign(budget=raw[FEES].sum(axis=1,min_count=5))
        variants={'raw':for_direct}
        for key in ('coarse','fine','displaced'):
            prob=z[key][:len(raw)]
            where=np.arange(len(raw))
            extra=pd.DataFrame(prob,columns=['fare_deck_'+d for d in DECKS])
            extra['registered_probability']=np.where(known,prob[where,label],np.nan)
            extra['maximum_deck_probability']=prob.max(1)
            extra.loc[~eligible,:]=np.nan
            variants[key]=pd.concat([for_direct,extra],axis=1)
    return variants


def train(arch):
    y,g,f,variants=restore_features()
    if arch=='neural':
        import hashlib
        import torch
        from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
        torch.set_num_threads(2)
    else:
        from solo_selection_likelihood_n26 import fit_model
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    for tag,ids,fold in [('historical',rows,hf),('full',np.arange(len(y)),f)]:
        for k in np.unique(fold):
            q=ids[fold==k];other=ids[fold!=k];t=other[~np.isin(g[other],g[q])]
            assert not np.isin(g[t],g[q]).any()
            path=OUT/f'{arch}_{tag}_{k}.npz'
            if not path.exists():
                preds={}
                for name,frame in variants.items():
                    if arch=='tree':
                        preds[name]=fit_model(frame.iloc[t],y[t]).predict_proba(frame.iloc[q])[:,1]
                    else:
                        preds[name]=infer(frame,y,t,q,5583+int(k))
                with path.with_suffix('.tmp').open('wb') as h:
                    np.savez_compressed(h,t=t,q=q,y=y[t],**preds)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for n,v in [('t',t),('q',q),('y',y[t])]:np.testing.assert_array_equal(z[n],v)
            print(arch,tag,k,flush=True)


def score():
    from scipy.special import logit
    from solo_archive_recovery_n26 import ticker
    from tabicl_consensus_review_n26 import sources, splits, OUT as REVIEW
    raw,y,g,panels=sources()
    report={'deck_prediction':json.loads((OUT/'deck_diagnostic.json').read_text())}
    booking=np.zeros(len(y),bool)
    on_review=np.zeros(len(y),bool)
    for key,src,dst,t,q in splits(panels,g):
        if key.startswith('bookingmix'):
            with np.load(REVIEW/f'pred_{key}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                booking[q]=True
                on_review[q]=(z['identities_error']>.5)&(z['identities_outcome']>.5)
    for tag,v in panels.items():
        ids,fold=v['ids'],v['fold']
        p={};anchor=v['anchor']
        for k in np.unique(fold):
            m=fold==k
            for arch in ('tree','neural'):
                with np.load(OUT/f'{arch}_{tag}_{k}.npz') as z:
                    np.testing.assert_array_equal(z['q'],ids[m])
                    for n in ('raw','coarse','fine','displaced'):
                        p.setdefault(arch+'_'+n,np.zeros(len(ids)))[m]=z[n]
        with np.load(OUT/'deck_estimates.npz') as z:
            eligible=z['eligible'][ids]
            valid=eligible&(z['label'][ids]>=0)
            disagree=valid&(z['fine'][ids].argmax(1)!=z['label'][ids])
        panels_out={}
        for ref_name,ref,coverage in [('parent214',anchor,v['known']),('review86',anchor^on_review[ids],v['known']&booking[ids])]:
            if tag=='historical' and ref_name=='review86':continue
            actions={}
            for arch in ('tree','neural'):
                raw_vote=p[arch+'_raw']>=.5
                for n in ('coarse','fine','displaced'):
                    after=p[arch+'_'+n]>=.5
                    actions[arch+'_'+n]=(after!=ref)
                    actions[arch+'_'+n+'_new']=(after!=ref)&(raw_vote==ref)
                native=logit(np.clip(p[arch+'_coarse'],.002,.998))
                step=logit(np.clip(p[arch+'_fine'],.002,.998))-native
                native_odds=v['off']*np.where(anchor,-1,1)
                actions[arch+'_increment']=((native_odds>=0)==ref)&((native_odds+step>=0)!=ref)
            actions['both_new']=actions['tree_fine_new']&actions['neural_fine_new']
            actions['one_new_both_confirm']=(actions['tree_fine_new']|actions['neural_fine_new'])&actions['tree_fine']&actions['neural_fine']
            wrong=ref!=y[ids]
            panels_out[ref_name]={n:{s:ticker(m&coverage,wrong,a&eligible) for s,m in [
                ('all',np.ones(len(ids),bool)),('unanimous',v['common']),('fare_disagrees_with_registered_deck',disagree)]}
                                for n,a in actions.items()}
            chosen = coverage & eligible & actions['neural_fine_new']
            delta = np.where(chosen, np.where(wrong, 1, -1), 0)
            order, position = np.unique(g[ids[coverage]], return_inverse=True)
            net = np.bincount(position, weights=delta[coverage])
            size = np.bincount(position)
            rng = np.random.default_rng(8841)
            estimates = []
            for _ in range(2000):
                sample = rng.integers(len(order), size=len(order))
                estimates.append(net[sample].sum() / size[sample].sum())
            panels_out[ref_name]['neural_fine_new']['paired_evaluation'] = dict(
                before=float((~wrong[coverage]).mean()),
                after=float((~wrong[coverage]).mean() + delta[coverage].mean()),
                net_by_fold=[int(delta[fold == k].sum()) for k in np.unique(fold)],
                booking_bootstrap_95=np.quantile(estimates,[.025,.975]).tolist())
        report[tag]=panels_out
        print(tag,[(n,a['all']['fixed'],a['all']['harmed'],a['unanimous']['fixed'],a['unanimous']['harmed'])
                   for n,a in panels_out['parent214'].items()],flush=True)
    RESULT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','tree','neural','score'])
    action=parser.parse_args().action
    if action=='prepare':prepare()
    elif action=='score':score()
    else:train(action)
