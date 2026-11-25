"""Check train/test separability and how domain reweighting changes paired candidate gains."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from joint_candidate_coverage_n26 import branch_decisions
from exaone_validation_data import compact_features, CATEGORICAL
from kaggle_workflow import sha256


def raw_features(frame):
    v=compact_features(frame)
    group=frame.PassengerId.str.split('_').str[0]
    v=v.assign(booking_number=pd.to_numeric(group),booking_size=group.map(group.value_counts()),
               booking_position=pd.to_numeric(frame.PassengerId.str.split('_').str[1]))
    return pd.get_dummies(v,columns=CATEGORICAL,dtype=float).to_numpy(float)


def score_features(p):
    quant=np.quantile(p,[0,.1,.25,.5,.75,.9,1],axis=1).T
    return np.column_stack([quant,p.mean(1),p.std(1),(p>=.5).sum(1)])


def propensity(x,domain,group):
    prob=np.full(len(domain),np.nan)
    ratio=prob.copy()
    auc=[]
    for train,query in GroupKFold(5,shuffle=True,random_state=2671).split(x,groups=group):
        assert not np.isin(group[train],group[query]).any()
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=9,min_samples_leaf=35,
            learning_rate=.06,l2_regularization=3,early_stopping=False,random_state=55)
        model.fit(x[train],domain[train])
        p=model.predict_proba(x[query])[:,1].clip(.00001,.99999)
        prob[query]=p
        rate=domain[train].mean()
        ratio[query]=p/(1-p)*(1-rate)/rate
        auc.append(float(roc_auc_score(domain[query],p)))
    assert np.isfinite(ratio).all()
    return ratio,dict(auc=float(roc_auc_score(domain,prob)),fold_auc=auc,log_loss=float(log_loss(domain,prob)))


def paired_effect(delta,w,group):
    n=len(w)
    w=w/w.mean()
    ids=np.unique(group,return_inverse=True)[1]
    num=np.bincount(ids,weights=w*delta)
    den=np.bincount(ids,weights=w)
    rng=np.random.default_rng(2672)
    means=[]
    for _ in range(20):
        ix=rng.integers(len(den),size=(100,len(den)))
        means.extend(100*num[ix].sum(1)/den[ix].sum(1))
    return dict(rows=n,weighted_fixed=float(w[delta>0].sum()),weighted_harmed=float(w[delta<0].sum()),
                equivalent_net_people=float((w*delta).sum()),gain_pp=float(100*np.average(delta,weights=w)),
                gain_pp_interval_fixed_weights=np.quantile(means,[.025,.975]).tolist(),
                effective_sample_size=float(n**2/(w@w)))


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,_,ref=panels_and_reference()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    freeze=ART/'proposal_n26_dispute_band/frozen_votes.npz'
    meta=json.loads((ROOT/'reports/proposal_n26_dispute_band/manifest.json').read_text())
    assert sha256(freeze)==meta['model_sha256']
    with np.load(freeze) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['names'],names)
        target=z['p']
    test_g=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(test_g,g).any()
    roster=pd.concat([raw,test],ignore_index=True)
    rawx=raw_features(roster)
    plans=[('competition',rawx,np.vstack([p,target]),np.r_[g,test_g],len(p)),
           ('same_passengers_old_vs_new',np.vstack([rawx[rows],rawx[rows]]),np.vstack([h,p[rows]]),np.tile(g[rows],2),len(rows))]
    report={}
    weights={'uniform':np.ones(len(raw))}
    for name,cov,opinions,group,boundary in plans:
        domain=np.arange(len(cov))>=boundary
        summary=score_features(opinions)
        result={}
        for method,x in [('raw',cov),('vote_distribution',summary),('individual_votes',opinions),
                         ('combined',np.column_stack([cov,opinions,summary]))]:
            w,status=propensity(x,domain,group)
            source=w[:boundary]
            status['source_weight_quantiles']=np.quantile(source,[0,.1,.5,.9,1]).tolist()
            status['clipped_fraction']=float(np.mean((source<.25)|(source>4)))
            result[method]=status
            if name=='competition':
                weights[method]=source.clip(.25,4)
            print(name,method,round(status['auc'],4),flush=True)
        report[name]=result
    # The unweighted gain reproduces the queue evaluation; only the measure on rows changes here.
    pred=branch_decisions(np.arange(len(y)),f,p,names,native,*ref,raw.CryoSleep.eq(False).to_numpy(),'full',891)
    known=ref[0]>=0
    effect={}
    for key in ('nine','band'):
        a=(pred['all_three']==y).astype(int)-(pred[key]==y)
        np.testing.assert_array_equal(a[0,known],a[1,known])
        assert a[0,known].sum()==(34 if key=='nine' else 19)
        effect[key]={}
        for name,w in weights.items():
            effect[key][name]=paired_effect(a[0,known],w[known],g[known])
            print(key,name,round(effect[key][name]['equivalent_net_people'],2),flush=True)
    report['target_weighted_effects']=effect
    report['method']=dict(weight_clip=[.25,4],folds=5,conditional_on_known_baseline=True,
        notes=['Weights rescaled to mean one on 8445 known passengers. Intervals resample bookings at fixed fitted weights.',
               'Same-person contrast changes both training size and training allocation. It is a reference for score-model effects.',
               'Target labels are unavailable. Weighting requires unchanged conditional model performance given the measured covariates.'])
    DEST.with_name('archive_domain_shift.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
