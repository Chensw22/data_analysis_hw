"""Estimate mistake odds by the source of a release decision and independent-versus-shared dissent."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from nested_release_review_n26 import ROOT
from evaluate_nested_release_n26 import DECK, OUT, REVIEW, SIX, outcome_scores, panel_predictions, protocols, replay, sources, stages, ticker


def choices(v,p,idx,enabled,risk,personal,score):
    anchor,review,fare,parent=stages(v,p,idx,enabled,risk)
    ids=v['ids'][idx]
    source=np.select([fare!=parent,review!=fare,anchor!=review],[3,2,1],0)
    family=(v['votes'][idx,:18]!=parent[:,None]).sum(1)
    vote=lambda names:np.column_stack([(p[n][idx]>=.5)!=parent for n in names]).sum(1)
    direction=np.where(parent,-1,1)
    native=v['off'][idx]*np.where(anchor==parent,1,-1)
    md=np.median(logit(score[idx].clip(.01,.99)),axis=1)*direction
    numeric=np.column_stack([native,md,direction,(v['common'][idx]).astype(float)])
    cells=pd.DataFrame(dict(direction=parent.astype(int),stage=source,
             family=np.digitize(family,[1,5,10,15]),registry=vote(SIX[:3]),receipt=vote(SIX[3:]),
             risk=np.digitize(expit(native),[.08,.20,.35,.50,.70]),
             center=np.digitize(expit(md),[.08,.20,.35,.50,.70]),cohort=v['common'][idx].astype(int),
             paid=personal[ids,0],party=personal[ids,1]))
    return parent,numeric,cells


def rates(columns,xt,xq,labels,pt,pq):
    prior=(labels.sum()+.5)/(len(labels)+1)
    simple=np.full(len(xq),prior)
    adjusted=pq.copy()
    for level in range(1,len(columns)+1):
        keys=columns[:level]
        table=xt[keys].assign(total=1.,mistake=labels.astype(float),expected=pt).groupby(keys,sort=False).sum()
        stats=xq[keys].join(table,on=keys)[['total','mistake','expected']].fillna(0.).to_numpy()
        n,y,expected=stats.T
        simple=(y+12*simple)/(n+12)
        # The finer cell replaces the effective number of observations from its parent.
        # Smooth the observed and expected counts toward the parent odds correction.
        adj=12*adjusted
        baseline=12*pq
        actual_rate=(y+adj)/(n+12)
        expected_rate=(expected+baseline)/(n+12)
        adjusted=expit(logit(pq)+logit(actual_rate.clip(.0001,.9999))-logit(expected_rate.clip(.0001,.9999)))
    return simple,adjusted


def run():
    raw,y,g,panels=sources()
    p={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    score={tag:outcome_scores(v,p[tag]) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
        agrees=(z['label']>=0)&(z['label']==z['fine'].argmax(1))
    fee=raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    personal=np.column_stack([np.where(enabled[:len(y)],2,fee.gt(0).any(axis=1).to_numpy().astype(int)),raw.booking_size.to_numpy().clip(max=3)])
    hierarchies=[['stage','risk','family','receipt'],['direction','family','registry','receipt'],
                 ['risk','stage','paid','registry'],['center','family','stage','party'],
                 ['receipt','registry','stage','risk'],['family','cohort','direction','risk']]
    bags={}
    for key,src,dst,t,q in protocols(panels,g):
        a,b=src['ids'][t],dst['ids'][q]
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],a)
            np.testing.assert_array_equal(z['outer'],b)
            sr=z['risk']
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            qr=np.column_stack([z['identities_error'],z['identities_outcome']])
        old,native_t,xt=choices(src,p['historical'],t,enabled[a],sr,personal,score['historical'])
        parent,native_q,xq=choices(dst,p['full'],q,enabled[b],qr,personal,score['full'])
        np.testing.assert_array_equal(parent,replay(key,q,dst,p['full'],enabled))
        wrong=old!=y[a]
        model=LogisticRegression(C=.3,max_iter=300).fit(native_t,wrong)
        pt=model.predict_proba(native_t)[:,1]
        pq=model.predict_proba(native_q)[:,1]
        rs=np.array([rates(cols,xt,xq,wrong,pt,pq) for cols in hierarchies])
        simple,adjusted=rs.mean(0)
        decisions={'prior':pq>.5,'pooled_cases':simple>.5,'adjusted_odds':adjusted>.5,
                   'both':(simple>.5)&(adjusted>.5),'both_new':(simple>.5)&(adjusted>.5)&(pq<=.5)}
        mode='bookingmix' if key.startswith('bookingmix') else key
        bags.setdefault(mode,[]).append((b,parent!=y[b],dst['common'][q],decisions))
    report={}
    for name,items in bags.items():
        ids,wrong,common=[np.concatenate([x[j] for x in items]) for j in range(3)]
        assert len(np.unique(ids))==len(ids)
        masks={'all':np.ones(len(ids),bool),'previous_unanimity':common,
               'paid_agree_unanimity':common&enabled[ids]&agrees[ids]}
        report[name]={n:{key:ticker(m,wrong,np.concatenate([x[3][n] for x in items])) for key,m in masks.items()} for n in items[0][3]}
        np.savez_compressed(OUT/f'cells_{name}.npz',ids=ids,wrong=wrong,
                            **{n:np.concatenate([x[3][n] for x in items]) for n in items[0][3]})
        print(name,[(k,v['all']['fixed'],v['all']['harmed']) for k,v in report[name].items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/historical_edit_cells.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
