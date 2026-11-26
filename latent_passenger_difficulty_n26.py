"""Model shared mistakes through latent passenger difficulty conditional on the proposed outcome."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp,softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs,summary
from export_budget_quarter_n23 import source
from export_joint_fare_foundations_n26 import NAME as RELEASE
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT,OUT as BASIC,data
from held_expert_portfolios_n26 import DECK,panel_predictions,sources
from kaggle_workflow import sha256
from nori_fare_party_n26 import ART as NORI,FEES
from six_way_against_scored_nine_n26 import test_decisions
from arbitration_against_scored_nine_n26 import champions

OUT=ROOT/'artifacts/nov26_latent_passenger_difficulty'
REPORT=ROOT/'reports/nov23_matched_learning_curve/latent_passenger_difficulty.json'
CUTS=np.array([.15,.35,.5,.65,.85])


def prepare():
    raw,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    truth,group,fold,_,names,p,_,mitra,_,_=panels_and_reference()
    for a,b in [(truth,y),(group,g),(fold,f)]:
        np.testing.assert_array_equal(a,b)
    art=ROOT/'artifacts'
    band='proposal_n26_dispute_band'
    bank=art/band/'frozen_votes.npz'
    assert sha256(bank)==json.loads((ROOT/'reports'/band/'manifest.json').read_text())['model_sha256']
    with np.load(bank) as z:
        np.testing.assert_array_equal(z['names'],names)
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        pred=np.vstack([p,z['p']])
    families=np.array([n.split('/')[0] for n in names])
    family_names=sorted(set(families))
    columns=[pred[:,families==n].mean(1) for n in family_names]
    with np.load(BASIC/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_allclose(z['p'],mitra,atol=1e-5,rtol=0)
    mt=source('proposal_online_first_counts15',test)[0]
    v=sources()[-1]['full']
    scores=panel_predictions(v,'full')
    with np.load(NORI/'aligned.npz') as z:
        for key,val in [('passenger',raw.PassengerId),('y',y),('fold',f)]:
            np.testing.assert_array_equal(z[key],val)
        nori=z['own']
        base=z['champion111']
    joined=art/RELEASE/'inputs.npz'
    assert sha256(joined)==json.loads((ROOT/'reports'/RELEASE/'manifest.json').read_text())['model_sha256']
    with np.load(joined) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        test_fine=z['fine']
    fine=np.column_stack([scores['neural_deck_fine'],nori])
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
    incumbent,known,_,hi=champions()
    # The fixed release only uses the fine probabilities; no comparison to the raw values is needed.
    act=eligible&(base==incumbent)&((fine>=.5)!=base[:,None]).all(1)
    queued=base^act
    assert ((queued==y)&known).sum()==7081 and known.sum()==8532
    qpred,_=test_decisions(RELEASE,test.PassengerId)
    columns.extend([np.r_[mitra,mt],np.r_[fine[:,0],test_fine[:,0]],np.r_[nori,test_fine[:,1]]])
    family_names.extend(['Mitra','TabICL fare','Nori fare'])
    frame=pd.concat([raw,test],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0]
    multi=groups.map(groups.value_counts()).to_numpy()>1
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    sleep=frame.CryoSleep.eq(True).to_numpy()
    regime=2*np.where(sleep,0,np.where(paid,1,2))+multi.astype(int)
    _,earlier,_=inputs()
    x=np.column_stack(columns)
    assert np.isfinite(x).all() and not np.isin(g,groups.iloc[len(y):]).any()
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'panel.npz',scores=x,names=family_names,regime=regime,
        passenger=frame.PassengerId.to_numpy(str),groups=groups.to_numpy(str),fold=f,y=y,
        anchor=np.r_[queued,qpred],known=known,common=earlier['common']&(earlier['parent']==queued),fresh=~np.isin(g,g[hi]))


def emissions(weight,indicator):
    prior=indicator.mean(0)
    return (weight.T@indicator.reshape(len(indicator),-1)).reshape(weight.shape[1],*prior.shape)+8*prior+.5


def conditional_class(indicator,context,k,correct_score):
    n=len(indicator)
    order=np.argsort(correct_score,kind='stable')
    w=np.full((n,k),.12/k)
    w[order,np.arange(n)*k//n]+=.88
    flat=indicator.reshape(n,-1)
    for _ in range(50):
        freq=emissions(w,indicator)
        lp=np.log(freq/freq.sum(2,keepdims=True)).reshape(k,-1)
        totals=np.eye(6)[context].T@w
        global_w=(w.sum(0)+1)/(n+k)
        mix=totals+12*global_w
        logmix=np.log(mix/mix.sum(1,keepdims=True))
        nxt=softmax(flat@lp.T+logmix[context],axis=1)
        if np.max(abs(w-nxt))<1e-5:
            break
        w=nxt
    return lp,logmix


def mixture(ordinal,regime,y,t,q,k):
    states=len(CUTS)+1
    onehot=np.eye(states)[ordinal]
    likelihood=[]
    for label in (False,True):
        ix=t[y[t]==label]
        aligned=ordinal[ix].mean(1)*(1 if label else -1)
        lp,mix=conditional_class(onehot[ix],regime[ix],k,aligned)
        evidence=onehot[q].reshape(len(q),-1)@lp.T
        joint_prior=(np.bincount(regime[ix],minlength=6)+2)/(np.bincount(regime[t],minlength=6)+4)
        likelihood.append(logsumexp(evidence+mix[regime[q]],axis=1)+np.log(joint_prior[regime[q]]))
    return expit(likelihood[1]-likelihood[0])


def run():
    with np.load(OUT/'panel.npz') as z:
        scores,regime,g,f,y,anchor,known,common,fresh=[z[k] for k in ['scores','regime','groups','fold','y','anchor','known','common','fresh']]
    ordinal=np.digitize(scores,CUTS)
    design=np.column_stack([np.eye(len(CUTS)+1)[ordinal].reshape(len(scores),-1),np.eye(6)[regime]])
    n=len(y)
    out={}
    for ordered in [False,True]:
        pred=np.full((n,3),np.nan)
        test=[]
        for outer in np.unique(f):
            ids=np.flatnonzero(f==outer)
            if ordered:
                blocks=np.array_split(np.unique(g[ids]),4)
                inner=[(ids[~np.isin(g[ids],b)],ids[np.isin(g[ids],b)]) for b in blocks]
            else:
                inner=[(ids[a],ids[b]) for a,b in GroupKFold(4,shuffle=True,random_state=1337+int(outer)).split(ids,groups=g[ids])]
            for t,q in [*inner,(ids,np.arange(n,len(scores)))]:
                assert not np.isin(g[t],g[q]).any()
                prob=np.column_stack([mixture(ordinal,regime,y,t,q,1),mixture(ordinal,regime,y,t,q,3),
                    LogisticRegression(C=.15,max_iter=500).fit(design[t],y[t]).predict_proba(design[q])[:,1]])
                if q[0]>=n:
                    test.append(prob)
                else:
                    assert np.isnan(pred[q]).all()
                    pred[q]=prob
        assert np.isfinite(pred).all() and np.isfinite(test).all()
        name='order_sorted' if ordered else 'shuffled'
        full=np.vstack([pred,np.mean(test,axis=0)])
        votes=(full>=.5)!=anchor[:,None]
        actions=dict(single=votes[:,0],difficulty=votes[:,1],ordinal_logistic=votes[:,2],
                     corroborated=votes[:,1]&votes[:,2],latent_only=votes[:,1]&~votes[:,0]&~votes[:,2])
        size=pd.Series(g[:n]).groupby(g[:n]).transform('size').to_numpy()
        masks=dict(all=known,common=common,fresh=known&fresh,solo=known&(size==1),
                   pair=known&(size==2),larger_booking=known&(size>=3),
                   zero_spend=pd.read_csv(ROOT/'data/raw/train.csv',usecols=FEES).gt(0).sum(axis=1).eq(0).to_numpy()&known)
        result={}
        for method,act in actions.items():
            result[method]={key:summary(act[:n],anchor[:n]!=y,m,f) for key,m in masks.items()}
            result[method]['test_edits']=int(act[n:].sum())
        result['matched_model_comparison']={key:summary((pred[:,0]>=.5)!=(pred[:,1]>=.5),(pred[:,0]>=.5)!=y,m,f) for key,m in masks.items()}
        out[name]=result
        np.savez_compressed(OUT/f'{name}.npz',p=full)
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    for name,record in out.items():
        print(name,[(k,v['all']['fixed'],v['all']['harmed'],v.get('test_edits')) for k,v in record.items()],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','run'])
    with threadpool_limits(1):
        if ap.parse_args().action=='prepare':
            prepare()
        else:
            run()
