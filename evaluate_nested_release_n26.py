"""Learn error and rollback policies from nested predictions of every stage of the current release."""
import json
import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from held_expert_portfolios_n26 import DECK, NEW, SIX, fare_model, outcome_scores, panel_predictions, physical, replay, six_model, sources, ticker
from nested_release_review_n26 import OUT, REVIEW, REPORT, protocols


def stages(v,p,idx,enabled,risk):
    anchor=v['anchor'][idx]
    review=anchor^((risk>.5).all(1))
    fare=fare_model(review,p['neural_deck_raw'][idx],p['neural_deck_fine'][idx],enabled)>=.5
    new=np.column_stack([p[n][idx] for n in SIX])
    final=six_model(fare,new,v['votes'][idx,:18],enabled)>=.5
    return anchor,review,fare,final


def features(v,p,s,idx,rawx,enabled,risk):
    ids=v['ids'][idx]
    anchor,review,fare,final=stages(v,p,idx,enabled,risk)
    direction=np.where(final,-1,1)
    odds=logit(s[idx].clip(.005,.995))*direction[:,None]
    target=odds[:,-len(NEW):]
    central=np.quantile(odds,[.1,.25,.5,.75,.9],axis=1).T
    shares=np.column_stack([(part>0).mean(1) for part in (odds[:,:18],target,odds)])
    native=v['off'][idx]*np.where(anchor==final,1,-1)
    base=np.column_stack([direction,native,central,shares,enabled,rawx[ids],target])
    flips=np.column_stack([anchor!=review,review!=fare,fare!=final])
    progression=np.column_stack([flips,anchor!=final,risk,risk.min(1),np.abs(risk[:,0]-risk[:,1])])
    trace=np.column_stack([base,progression,flips*native[:,None]])
    return base,trace,final,shares


def fit_risk(x,y,test):
    mean=x.mean(0)
    scale=x.std(0).clip(.2)
    tr=((x-mean)/scale).clip(-6,6)
    te=((test-mean)/scale).clip(-6,6)
    lr=LogisticRegression(C=.06,max_iter=700).fit(tr,y)
    gb=HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=6,min_samples_leaf=35,learning_rate=.04,
                                      l2_regularization=6,early_stopping=False,random_state=961).fit(tr,y)
    return lr.predict_proba(te)[:,1],gb.predict_proba(te)[:,1]


def run():
    raw,y,g,panels=sources()
    rawx=physical(raw)
    p={tag:panel_predictions(v,tag) for tag,v in panels.items()}
    s={tag:outcome_scores(v,p[tag]) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
        agrees=(z['label']>=0)&(z['fine'].argmax(1)==z['label'])
    collected={}
    for key,src,dst,t,q in protocols(panels,g):
        a,b=src['ids'][t],dst['ids'][q]
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],a)
            np.testing.assert_array_equal(z['outer'],b)
            nested=z['risk']
            for k in np.unique(z['fold']):
                assert not np.isin(g[a[z['fold']==k]],g[a[z['fold']!=k]]).any()
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            outer=np.column_stack([z['identities_error'],z['identities_outcome']])
        bx,xt,parent,_=features(src,p['historical'],s['historical'],t,rawx,enabled[a],nested)
        bq,xq,current,shares=features(dst,p['full'],s['full'],q,rawx,enabled[b],outer)
        np.testing.assert_array_equal(current,replay(key,q,dst,p['full'],enabled))
        target=parent!=y[a]
        controls=src['anchor'][t]!=y[a]
        predictions={}
        for mode,train,query,labels in [('proxy_error',bx,bq,controls),('nested_error',bx,bq,target),
                                      ('nested_trace',xt,xq,target)]:
            rl,rg=fit_risk(train,labels,query)
            predictions[mode+'_ridge']=rl
            predictions[mode+'_tree']=rg
        minrisk=np.minimum(predictions['nested_trace_ridge'],predictions['nested_trace_tree'])
        predictions['two_reviewers']=minrisk
        flip={n:pr>.5 for n,pr in predictions.items()}
        flip['rollback_only']=(minrisk>.5)&(current!=dst['anchor'][q])
        flip['additional_only']=(minrisk>.5)&(current==dst['anchor'][q])
        flip['supported_by_new_experts']=(minrisk>.5)&(shares[:,1]>.5)
        tag='bookingmix' if key.startswith('bookingmix') else key
        collected.setdefault(tag,[]).append((b,current!=y[b],dst['common'][q],flip,predictions,
                                            int((controls!=target).sum()),len(a)))
    report={}
    for tag,items in collected.items():
        ids,wrong,common=[np.concatenate([x[i] for x in items]) for i in range(3)]
        assert len(np.unique(ids))==len(ids)
        masks={'all':np.ones(len(ids),bool),'previous_unanimity':common,'paid_agree_unanimity':common&enabled[ids]&agrees[ids]}
        r={}
        for n in items[0][3]:
            act=np.concatenate([x[3][n] for x in items])
            r[n]={k:ticker(m,wrong,act) for k,m in masks.items()}
            if n in items[0][4]:
                prob=np.concatenate([x[4][n] for x in items])
                r[n]['error_auc']=float(roc_auc_score(wrong,prob))
        r['source_label_changes_by_fit']=[{'changed':x[5],'source_rows':x[6]} for x in items]
        report[tag]=r
        print(tag,[(k,v['all']['fixed'],v['all']['harmed']) for k,v in r.items() if isinstance(v,dict)],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
