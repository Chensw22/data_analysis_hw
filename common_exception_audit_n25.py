"""Rank unanimous errors on disjoint orders and count false alarms in the same cohort."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from disjoint_tree_fields_n25 import ART, DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets

BILLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def inputs(raw):
    cabin = raw.Cabin.str.split('/', expand=True)
    registration = raw[['HomePlanet','CryoSleep','Destination','VIP']].copy()
    registration['deck'], registration['side'] = cabin[0], cabin[2]
    registration['age'] = pd.cut(raw.Age, [-1,12,17,25,40,60,120])
    registration['cabin_band'] = np.floor(pd.to_numeric(cabin[1])/150)
    orders = raw.PassengerId.str.split('_').str[0]
    registration['party'] = orders.map(orders.value_counts()).clip(upper=4)
    expenses = pd.DataFrame(index=raw.index)
    for name in BILLS:
        expenses[name] = pd.cut(raw[name], [-1,0,50,250,750,1500,3000,np.inf])
    expenses['positive_count'] = raw[BILLS].gt(0).sum(axis=1)
    expenses['total'] = pd.cut(raw[BILLS].sum(axis=1,min_count=1),[-1,0,50,250,750,1500,3000,np.inf])
    def encode(df):
        return pd.get_dummies(df.astype('string').fillna('?'),dtype=float).to_numpy()
    return encode(registration), encode(expenses)


def rank(train, test, side, features, mistakes, permuted, seed):
    scores = np.empty(len(test))
    for s in [False,True]:
        a = train[side[train]==s]
        mask = side[test]==s
        target = mistakes[a].copy()
        if permuted:
            np.random.default_rng(seed+int(s)).shuffle(target)
        if features is None or len(np.unique(target))<2:
            scores[mask] = (target.sum()+1)/(len(target)+2)
        else:
            model = LogisticRegression(C=.12,max_iter=300,solver='liblinear').fit(features[a],target)
            scores[mask] = model.predict_proba(features[test[mask]])[:,1]
    return scores


def evaluate(raw,archive,y,g,fold,train_hist,arch_hist):
    full = np.column_stack(archive)
    hist = np.column_stack(arch_hist)
    n = len(raw)
    union = np.vstack([full, hist])
    sign = union[:,0]>=.5
    same = ((union>=.5)==sign[:,None]).all(axis=1)
    wrong = sign!=np.r_[y,y[train_hist]]
    # Build unlabeled covariates once on the original roster, preserve passenger IDs across protocols.
    reg,pay = inputs(raw)
    reg,pay = (np.vstack([x,x[train_hist]]) for x in (reg,pay))
    directed=np.where(sign[:,None],1-union,union)
    confidence=np.quantile(directed,[0,.25,.5,.75,1],axis=1).T*5
    scopes={'within_outer_fold':[],'historical_to_new_orders':[]}
    for k in np.unique(fold):
        outer = np.flatnonzero((fold==k)&same[:n])
        for a,b in GroupKFold(4,shuffle=True,random_state=137).split(outer,groups=g[outer]):
            t,q=outer[a],outer[b]
            assert not np.isin(g[t],g[q]).any()
            scopes['within_outer_fold'].append((t,q))
    fresh = ~np.isin(g,g[train_hist])
    t,q=np.flatnonzero(same[n:])+n,np.flatnonzero(same[:n]&fresh)
    assert not np.isin(g[train_hist[t-n]],g[q]).any()
    scopes['historical_to_new_orders'].append((t,q))
    report={}
    for scope, splits in scopes.items():
        report[scope]={}
        for name,feature in [('direction_only',None),('registration',reg),('payment',pay),
                ('both',np.column_stack([reg,pay])),('confidence',confidence),
                ('both_and_confidence',np.column_stack([reg,pay,confidence])),
                ('shuffle_within_direction',np.column_stack([reg,pay,confidence]))]:
            ids=[]; risk=[]; selections=[]
            for i,(train,test) in enumerate(splits):
                p=rank(train,test,sign,feature,wrong,name=='shuffle_within_direction',163+i)
                # Order ties by a fixed label-blind permutation, including in the direction-only control.
                tie=np.random.default_rng(807+i).permutation(len(test))
                order=tie[np.argsort(-p[tie],kind='stable')]
                rank_pos=np.argsort(order)
                within_direction=np.zeros((len(test),2),bool)
                for side in (False,True):
                    side_order=order[sign[test[order]]==side]
                    for j,fraction in enumerate((.01,.05)):
                        within_direction[side_order[:max(1,round(fraction*len(side_order)))],j]=True
                selections.append(np.column_stack([p>=.5,rank_pos<max(1,round(.01*len(test))),
                                                    rank_pos<max(1,round(.05*len(test))),within_direction]))
                ids.extend(test);risk.extend(p)
            ix=np.array(ids);risk=np.array(risk);chosen=np.vstack(selections)
            assert len(np.unique(ix))==len(ix)
            def metrics(m):
                target=wrong[ix[m]]
                return dict(passengers=len(target),errors=int(target.sum()),auc=float(roc_auc_score(target,risk[m])),
                           average_precision=float(average_precision_score(target,risk[m])),
                           flips={k:dict(fixed=int(np.sum(a&target)),harmed=int(np.sum(a&~target)))
                                  for k,a in zip(['risk_half','top_one_percent','top_five_percent',
                                                  'within_direction_top_one','within_direction_top_five'],chosen[m].T)})
            report[scope][name]={k:metrics(m) for k,m in [('all',np.ones(len(ix),bool)),
                                 ('unanimous_false',~sign[ix]),('unanimous_true',sign[ix])]}
    return report


def run():
    raw,y,g,fold=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,fold,rows,hf)
    report=evaluate(raw,panels[0],y,g,fold,rows,panels[1])
    info=dict(training='Separate regularized logistic regression for each unanimous direction, C=0.12.',
              validation='Four group partitions inside each original outer fold; separately train on all historical unanimous cases and query only previously unseen orders.',
              target='68/68 agreement; mistake means their shared label differs from truth. Tail fractions measure retrieval at fixed workload, not deployment recommendations.',
              results=report)
    DEST.with_name('common_exception_audit_n25.json').write_text(json.dumps(info,indent=2)+'\n')
    for scope,r in report.items():
        for name,v in r.items():
            m=v['all']
            print(scope,name,round(m['auc'],3),m['flips'],flush=True)


if __name__=='__main__':
    run()
