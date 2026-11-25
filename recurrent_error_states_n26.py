"""Predict original and repeated failures jointly, with auxiliary labels from disjoint reference bookings."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from booking_mix_confidence_transfer_n26 import assignments, partition
from exaone_validation_data import CATEGORICAL
from full_roster_booking_validation_n23 import ROOT
from tabicl_consensus_review_n26 import OUT as REVIEW, matrices, sources

ART = ROOT/'artifacts/nov26_recurrent_error_states'
REPORT = ROOT/'reports/nov23_matched_learning_curve/recurrent_error_states.json'


def match_counts(old, cur, group):
    ids = old['ids']
    known = old['known'] & cur['known'][ids]
    current, historical = cur['wrong'][ids], old['wrong']
    history_groups = group[ids]
    cold = np.zeros(len(ids), bool)
    for f in np.unique(old['fold']):
        mask = old['fold']==f
        cold[mask] = ~np.isin(history_groups[mask], history_groups[~mask])
    both = old['common'] & cur['common'][ids]
    out = {}
    for name, mask in [('all',known),('current_unanimous',known&cur['common'][ids]),
                      ('both_unanimous',known&both),('cold_both_unanimous',known&both&cold)]:
        out[name] = dict(rows=int(mask.sum()),both_wrong=int((mask&current&historical).sum()),
                         current_only_wrong=int((mask&current&~historical).sum()),
                         historical_only_wrong=int((mask&~current&historical).sum()),
                         both_correct=int((mask&~current&~historical).sum()))
    out['current_unanimous_errors_without_known_history'] = int((cur['common']&cur['wrong']).sum() -
                                                                 out['current_unanimous']['both_wrong'] -
                                                                 out['current_unanimous']['current_only_wrong'])
    return out


def auxiliary():
    from catboost import CatBoostClassifier
    raw, y, g, v = sources()
    old=v['historical']
    ART.mkdir(exist_ok=True)
    for f in np.unique(old['fold']):
        held=old['ids'][old['fold']==f]
        alltrain=old['ids'][old['fold']!=f]
        train=alltrain[~np.isin(g[alltrain],g[held])]
        half=partition(g[train],'auxiliary-halves:',2)
        assert not np.isin(g[train[half==0]],g[train[half==1]]).any()
        wrong=[]
        for j in range(2):
            learn=train[half==j]
            assert not np.isin(g[learn],g[held]).any()
            estimator=CatBoostClassifier(iterations=450,depth=5,learning_rate=.055,l2_leaf_reg=6,
                    verbose=False,allow_writing_files=False,thread_count=2,cat_features=CATEGORICAL,random_seed=1741+j)
            estimator.fit(raw.iloc[learn],y[learn])
            guess=estimator.predict_proba(raw.iloc[held])[:,1]>=.5
            wrong.append(guess!=y[held])
        np.savez_compressed(ART/f'aux{f}.npz',held=held,train=train,half=half,wrong=np.stack(wrong,axis=1))


def learn():
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import DIGEST, WEIGHT
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    for f,k,t,q in assignments(old,cur,g):
        path=ART/f'fit{f}_{k}.npz'
        if path.exists():
            continue
        with np.load(ART/f'aux{f}.npz') as z:
            assert not np.isin(g[z['train']],g[q]).any()
            lookup=pd.Index(z['held']).get_indexer(old['ids'][t])
            assert (lookup>=0).all()
            count=z['wrong'][lookup].sum(1)
        wrong=old['wrong'][t].astype(int)
        strata=4*wrong+2*old['anchor'][t]+old['common'][t]
        rng=np.random.default_rng(6253+5*f+k)
        sham=count.copy()
        for c in np.unique(strata):
            loc=np.flatnonzero(strata==c)
            sham[loc]=rng.permutation(count[loc])
        xt,xq=matrices(raw,old,cur,t,q)
        xt=xt[1].assign(unanimous=old['common'][t].astype(int))
        xq=xq[1].assign(unanimous=cur['common'][q].astype(int))
        fields={}
        for name,label in [('binary',wrong),('repeat',3*wrong+count),('shuffled',3*wrong+sham)]:
            est=model(8813).fit(xt,label)
            pred=est.predict_proba(xq)
            categories=est.classes_.astype(int)
            if name=='binary':
                fields['binary']=pred[:,categories==1].ravel()
            else:
                fields[name]=pred[:,categories>=3].sum(1)
                fields[name+'_two_or_more']=pred[:,categories//3+categories%3>=2].sum(1)
                fields[name+'_all']=pred[:,categories==5].sum(1)
        np.savez_compressed(path,t=old['ids'][t],q=q,truth=y[old['ids'][t]],**fields)
        print(f,k,flush=True)


def actual_pair_learn():
    import torch
    from joint_party_tabicl_n26 import model
    from tabicl_routed_context_n23 import WEIGHT,DIGEST
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    latest=cur['ids']
    prior=old['ids']
    fresh=~np.isin(g,g[prior])
    # Historical model outputs contain no label from any of these entirely new orders.
    # Use current scores only from the query's held fold, for both support and query.
    for k in np.unique(cur['fold']):
        mask=old['known']&cur['known'][prior]&(cur['fold'][prior]==k)
        t=prior[mask]
        q=latest[cur['known']&fresh&(cur['fold']==k)]
        assert not np.isin(g[t],g[q]).any() and not np.isin(g[prior],g[q]).any()
        path=ART/f'paired{k}.npz'
        xt,xq=matrices(raw,cur,cur,t,q)
        xt=xt[1].assign(unanimous=cur['common'][t].astype(int))
        xq=xq[1].assign(unanimous=cur['common'][q].astype(int))
        wrong=cur['wrong'][t].astype(int)
        before=old['wrong'][mask].astype(int)
        shuffled=before.copy()
        strata=4*wrong+2*cur['anchor'][t]+cur['common'][t]
        rng=np.random.default_rng(156+k)
        for c in np.unique(strata):
            loc=np.flatnonzero(strata==c)
            shuffled[loc]=rng.permutation(before[loc])
        estimates={}
        for name,label in [('binary',wrong),('paired',2*wrong+before),('sham',2*wrong+shuffled)]:
            learner=model(4528).fit(xt,label)
            prob=learner.predict_proba(xq)
            cls=learner.classes_.astype(int)
            if name=='binary':
                estimates[name]=prob[:,cls==1].ravel()
            else:
                estimates[name]=prob[:,cls>=2].sum(1)
                estimates[name+'_persistent']=prob[:,cls==3].sum(1)
        np.savez_compressed(path,t=t,q=q,**estimates)
        print('paired',k,flush=True)


def actual_pair_score(cur,old,g):
    tested=cur['known']&~np.isin(g,g[old['ids']])
    probabilities={}
    for k in np.unique(cur['fold']):
        use=tested&(cur['fold']==k)
        with np.load(ART/f'paired{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],np.flatnonzero(use))
            assert (cur['fold'][z['t']]==k).all()
            for name in ['binary','paired','paired_persistent','sham','sham_persistent']:
                probabilities.setdefault(name,np.zeros(len(g)))[use]=z[name]
    out={}
    for name,p in probabilities.items():
        table={}
        for group,scope in [('new',tested),('new_common',tested&cur['common'])]:
            w=cur['wrong'][scope]
            flip=p[scope]>.5
            table[group]=dict(rows=len(w),errors=int(w.sum()),fixed=int((flip&w).sum()),
                harmed=int((flip&~w).sum()),auc=float(roc_auc_score(w,p[scope])))
        out[name]=table
    return out


def score():
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    report={'overlap':match_counts(old,cur,g)}
    repeats=np.full(len(old['ids']),-1)
    for f in np.unique(old['fold']):
        ix=old['fold']==f
        with np.load(ART/f'aux{f}.npz') as z:
            np.testing.assert_array_equal(old['ids'][ix],z['held'])
            repeats[ix]=z['wrong'].sum(1)
    cohort=old['known']&old['common']
    report['auxiliary_training_agreement']={}
    for name,w in [('historical_wrong',old['wrong']),('historical_correct',~old['wrong'])]:
        report['auxiliary_training_agreement'][name]=np.bincount(repeats[cohort&w],minlength=3).tolist()
    table={}
    reviewed=np.zeros(len(y),bool)
    tested=np.zeros(len(y),bool)
    for f,k,t,q in assignments(old,cur,g):
        with np.load(ART/f'fit{f}_{k}.npz') as z:
            np.testing.assert_array_equal(z['t'],old['ids'][t])
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['truth'],y[old['ids'][t]])
            for n in ('binary','repeat','repeat_two_or_more','repeat_all','shuffled','shuffled_two_or_more','shuffled_all'):
                table.setdefault(n,np.zeros(len(y)))[q]=z[n]
        with np.load(REVIEW/f'pred_bookingmix_{f}_{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            reviewed[q]=(z['identities_error']>.5)&(z['identities_outcome']>.5)
        assert not tested[q].any()
        tested[q]=True
    fresh=~np.isin(g,g[old['ids']])
    for n,p in table.items():
        out={}
        for cohort,mask in [('all',tested),('common',tested&cur['common']),('new',tested&fresh),
                            ('new_common',tested&fresh&cur['common']),('outside_latest_review',tested&~reviewed)]:
            actual=cur['wrong'][mask]
            flag=p[mask]>.5
            top=np.argsort(-p[mask],kind='stable')[:int(np.ceil(len(actual)*.1))]
            out[cohort]=dict(rows=len(actual),errors=int(actual.sum()),fixed=int((actual&flag).sum()),
                             harmed=int((~actual&flag).sum()),auc=float(roc_auc_score(actual,p[mask])),
                             top_tenth_rows=len(top),top_tenth_errors=int(actual[top].sum()))
            if n in ('binary','repeat','shuffled'):
                out[cohort]['nll']=float(log_loss(actual,p[mask]))
        report[n]=out
    report['actual_states_on_new_orders']=actual_pair_score(cur,old,g)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['overlap'])
    print([(n,x['all']['fixed'],x['all']['harmed'],x['common']['fixed'],x['common']['harmed'],x['new']['fixed'],x['new']['harmed']) for n,x in report.items() if 'all' in x and n!='overlap'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['auxiliary','learn','pair','score'])
    {'auxiliary':auxiliary,'learn':learn,'pair':actual_pair_learn,'score':score}[parser.parse_args().action]()
