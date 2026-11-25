"""Score both hypothetical labels by their effect on independently held-out historical bookings."""
import json

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import assignments
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, ROOT, matrices, sources


def features(a,b):
    a,b=a.copy(),b.copy()
    fees=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for df in (a,b):
        df[fees]=np.log1p(df[fees])
    a=pd.get_dummies(a,dtype=float)
    b=pd.get_dummies(b,dtype=float).reindex(columns=a.columns,fill_value=0)
    return np.vstack([d.fillna(-1).to_numpy(float) for d in (a,b)])


def mapping(x,fit,seed):
    scale=x[fit].std(0).clip(.25)
    z=((x-x[fit].mean(0))/scale).clip(-5,5)
    rng=np.random.default_rng(seed)
    left,right=rng.choice(fit,(2,500))
    width=np.sqrt(np.median(np.sum((z[left]-z[right])**2,axis=1))).clip(1)
    direction=rng.normal(size=(x.shape[1],96))/width
    rff=np.sqrt(2/96)*np.cos(z@direction+rng.uniform(0,2*np.pi,96))
    return np.column_stack([np.ones(len(z)),.45*z/np.sqrt(z.shape[1]),rff])


def probe_scores(phi,fit,probe,q,offset,target,permuted,verify):
    a,z,b=phi[fit],phi[probe],phi[q]
    gram=a.T@a+1.5*np.eye(a.shape[1])
    rhs=a.T@(target[fit]-offset[fit])
    factor=cho_factor(gram)
    beta=cho_solve(factor,rhs)
    h=cho_solve(factor,b.T)
    probe_base=offset[probe]+z@beta
    own=offset[q]+b@beta
    # Rank-one inverse for adding the query with either binary label at unit weight.
    influence=(z@h)/(1+(b*h.T).sum(1))
    alternatives=[probe_base[:,None]+influence*(candidate-own) for candidate in (0,1)]
    if verify:
        for label in (0,1):
            u=b[0]
            after=np.linalg.solve(gram+np.outer(u,u),rhs+u*(label-offset[q[0]]))
            np.testing.assert_allclose(alternatives[label][:,0],offset[probe]+z@after,atol=1e-11)
    p0,p1=(p.clip(.015,.985) for p in alternatives)
    yes,no=np.log(p1/p0),np.log((1-p1)/(1-p0))
    real=np.where(target[probe,None],yes,no)
    fake=np.where(permuted[:,None],yes,no)
    donor_residual=target[probe]-probe_base
    updated=own+influence.T@donor_residual
    return dict(support=own.clip(.005,.995),direct_residual=updated.clip(.005,.995),
                measured=real.sum(0),permuted=fake.sum(0))


def evaluate(prob,ref,wrong,scope,mass):
    ids=np.flatnonzero(scope)
    pred=prob>.5
    top=ids[np.argsort(-prob[ids],kind='stable')[:int(np.ceil(.1*len(ids)))]]
    delta=(pred!=ref)*np.where(wrong,1,-1)
    return dict(**ticker(scope,wrong,pred!=ref),auc=float(roc_auc_score((wrong^ref)[ids],prob[ids])),
                nll=float(log_loss((wrong^ref)[ids],prob[ids])),top_tenth=len(top),top_tenth_errors=int((wrong^ref)[top].sum()),
                gain_test_mix_pp=float(100*np.average(delta[ids],weights=mass[ids])))


def run():
    raw,y,g,v=sources()
    old,cur=v['historical'],v['full']
    fresh=~np.isin(g,g[old['ids']])
    records={};tested=np.zeros(len(y),bool)
    previous=np.zeros(len(y),bool)
    for f,k,t,q in assignments(old,cur,g):
        train_x,test_x=matrices(raw,old,cur,t,q)
        with np.load(REVIEW/f'pred_bookingmix_{f}_{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            previous[q]=(z['identities_error']>.5)&(z['identities_outcome']>.5)
        groups=g[old['ids'][t]]
        offset=expit(np.r_[old['off'][t],cur['off'][q]])
        truth=old['wrong'][t]
        cells=old['anchor'][t].astype(int)*6+old['common'][t]*3+np.digitize(offset[:len(t)],[.12,.30])
        splits=list(GroupKFold(3,shuffle=True,random_state=7603).split(t,groups=groups))
        for name,left,right in [('physical',train_x[0].iloc[:,:raw.shape[1]],test_x[0].iloc[:,:raw.shape[1]]),
                                ('with_confidence',train_x[1],test_x[1])]:
            x=features(left,right)
            folds=[]
            for j,(base,probe) in enumerate(splits):
                assert not np.isin(groups[base],groups[probe]).any()
                rng=np.random.default_rng(8602+int(f)*15+k*3+j)
                fake=truth[probe].copy()
                for c in np.unique(cells[probe]):
                    match=np.flatnonzero(cells[probe]==c)
                    fake[match]=rng.permutation(fake[match])
                phi=mapping(x,base,756+j)
                folds.append(probe_scores(phi,base,probe,len(t)+np.arange(len(q)),offset,truth,fake,f==0 and k==0))
            avg={n:np.mean([r[n] for r in folds],axis=0) for n in folds[0]}
            for n,score in [('support',avg['support']),('direct',avg['direct_residual']),
                            ('actual',expit(logit(avg['support'])+avg['measured'])),
                            ('shuffled',expit(logit(avg['support'])+avg['permuted']))]:
                records.setdefault(name+'_'+n,np.zeros(len(y)))[q]=score
        tested[q]=True
    assert tested.sum()==6055
    mix=json.loads((ROOT/'reports/nov23_matched_learning_curve/booking_mix_confidence_transfer.json').read_text())
    band=raw.booking_size.to_numpy().clip(max=4)-1
    mass=np.array(mix['target_test_share'])[band]/np.bincount(band[tested])[band]
    wrong=cur['wrong']
    report={}
    scopes={'all':tested,'common':tested&cur['common'],'fresh':tested&fresh}
    for name,prob in records.items():
        report[name]={}
        for base,ref in [('parent214',np.zeros(len(y),bool)),('review86',previous)]:
            report[name][base]={s:evaluate(prob,ref,wrong^ref,sel,mass) for s,sel in scopes.items()}
    for name,prob in records.items():
        if not name.endswith('_actual'): continue
        initial=records[name.replace('_actual','_support')]
        induced=(prob>.5)&(initial<=.5)
        report[name]['new_vs_support']={s:ticker(m,wrong,induced) for s,m in scopes.items()}
    print([(n,a['parent214']['all']['fixed'],a['parent214']['all']['harmed'],a['parent214']['common']['fixed'],a['parent214']['common']['harmed'])
           for n,a in report.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/counterlabel_probe_likelihood.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
