"""Compare independent TabNet and TabDPT opinions on the scored directional release."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
OUT=ART/'nov27_outsider_directional'
REPORT=ROOT/'reports/nov23_matched_learning_curve/outsider_directional_n27.json'


def test_dpt():
    import torch
    import tabdpt_adapter as adapter
    import tabdpt_resource_probe as backend
    from exaone_validation_data import compact_features
    assert torch.backends.mps.is_available()
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    adapter.verify_sources()
    raw=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda k:k!='Name')
    y=raw.pop('Transported').to_numpy(int)
    tst=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    assert not np.isin(tst.PassengerId.str.split('_').str[0],raw.PassengerId.str.split('_').str[0]).any()
    x,query=compact_features(raw),compact_features(tst)
    enc=backend.CompactEncoder().fit(x)
    model=backend.new_model('mps')
    model.fit(enc.transform(x),y)
    p=backend.probabilities(model,enc.transform(query),ensembles=2,seed=899)[:,1]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'tabdpt.npz',ids=tst.PassengerId.to_numpy(str),train=raw.PassengerId.to_numpy(str),y=y,p=p,sha=backend.WEIGHT_SHA)


def report():
    from full_roster_booking_validation_n23 import data
    from tabnet_attention_reliability_n26 import signals,network_outputs
    from repeated_archive_misses_n26 import cohort,count
    from directional_foundation_transfer_n27 import calibrate
    from booking_mix_confidence_transfer_n26 import partition
    from export_directional_foundation_n27 import NAME as PARENT
    from rebase_fare_six_n26 import NAME as BASE
    from six_way_against_scored_nine_n26 import test_decisions
    from scipy.special import expit,logit
    raw,y,g,f=data()
    _,blind,known,fresh=cohort(raw,y,f)
    n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv').PassengerId
    ids=np.r_[raw.PassengerId.to_numpy(str),test.to_numpy(str)]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        foundations=z['scores'][:,-3:]
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        first=np.r_[z['champion111'],test_decisions(BASE,test)[0]].astype(bool)
    parent=np.r_[first[:n],test_decisions(PARENT,test)[0]]
    med=np.median(foundations,axis=1)
    risk=np.where(first,1-med,med)
    split=partition(g,'directional_foundation',4)
    for k in range(8):
        for j in range(4):
            t=np.flatnonzero(known&(f==k)&(split!=j))
            q=np.flatnonzero((f==k)&(split==j))
            cut=calibrate(risk,first,y,t,1.)
            parent[q]=first[q]^(risk[q]>=cut[first[q].astype(int)])
    frozen=first!=parent
    assert frozen[n:].sum()==16
    net,_=signals(raw,y,g,f)
    with np.load(ART/'nov26_tabnet/test.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test)
        net=np.vstack([net,network_outputs(z)[0]])
    dpt=np.empty(len(ids))
    with np.load(OUT/'tabdpt.npz') as z:
        np.testing.assert_array_equal(z['ids'],test)
        np.testing.assert_array_equal(z['train'],raw.PassengerId)
        np.testing.assert_array_equal(z['y'],y)
        sha=z['sha'].item()
        dpt[n:]=z['p']
    for k in range(8):
        with np.load(ART/f'nov25_cross_architecture_state/{k}_global.npz') as z:
            q=np.flatnonzero(f==k)
            t=np.flatnonzero(f!=k)
            for key,val in [('passenger',raw.PassengerId.iloc[q]),('q',q),('t',t),('training_labels',y[t]),('checkpoint',sha)]:
                np.testing.assert_array_equal(z[key],val)
            dpt[q]=z['prob']
    pf=np.where(first[:,None],1-foundations,foundations)
    of=np.where(first,1-dpt,dpt)
    nf=np.where(first,1-net.mean(1),net.mean(1))
    nv=(pf>.5).sum(1)
    signed=lambda p:logit(p.clip(.002,.998))
    votes={'TabNet':nf,'TabDPT':of,'two_min':np.minimum(nf,of),'two_mean':expit((signed(nf)+signed(of))/2),
           'one_with_foundation':np.minimum(np.maximum(nf,of),pf.max(1)),
           'two_with_foundation':np.minimum(np.minimum(nf,of),pf.max(1)),
           'foundation_control':np.median(pf,axis=1)}
    order=np.zeros(n,int)
    random=partition(g,'independent_review',4)
    for k in range(8):
        for j,group in enumerate(np.array_split(np.unique(g[f==k]),4)):
            order[(f==k)&np.isin(g,group)]=j
    report={}
    outputs={}
    for key,score in votes.items():
        for region,mask in [('unanimous',nv==0),('split',(nv>0)&(nv<3)),('retained',np.ones(len(ids),bool))]:
            for scheme,bins in [('random',random),('ordered',order)]:
                pred=np.zeros(len(ids),bool)
                tests=[]
                for k in range(8):
                    for j in [0,1,2,3,None]:
                        t=np.flatnonzero(known & mask[:n] & (f==k)&((bins!=j) if j is not None else True))
                        q=np.flatnonzero((f==k)&(bins==j)) if j is not None else np.arange(n,len(ids))
                        # The original parent's 16 positive edits are preserved. Only raw-score opposition is calibrated.
                        cut=calibrate(score,first,y,t,1.)
                        a=mask[q]&~frozen[q]&(score[q]>=cut[first[q].astype(int)])
                        if j is None:tests.append(a)
                        else:pred[q]=a
                pred[n:]=np.mean(tests,axis=0)>.5
                tag='_'.join([key,region,scheme])
                outputs[tag]=pred
                assert not np.any(pred&frozen)
                delta=pred[:n]*np.where(parent[:n]!=y,1,-1)
                report[tag]=dict(test=int(pred[n:].sum()),fold_net=[int(delta[known&(f==k)].sum()) for k in range(8)],
                                 scopes={s:count(pred[:n],m,parent[:n]!=y) for s,m in [('all',known),('fresh',fresh&known),('common31',blind)]})
    from scipy.optimize import minimize
    log=lambda a:logit(a.clip(.005,.995))
    core=np.column_stack([np.ones(len(ids)),log(foundations)])
    native=log(med)
    extern=np.column_stack([log(dpt),log(net.mean(1))])
    # Fit the class outcome on independent bookings, without exposing a calibrated release label in training.
    for label,x in [('control',core),('new',np.column_stack([core,extern])),
                   ('spread',np.column_stack([core,extern,net.std(1),(foundations>=.5).mean(1)]))]:
        for scheme,bins in [('random',random),('ordered',order)]:
            value=np.empty(len(ids))
            pack=[]
            for k in range(8):
                for j in [0,1,2,3,None]:
                    t=np.flatnonzero((f==k)&((bins!=j) if j is not None else True))
                    q=np.flatnonzero((f==k)&(bins==j)) if j is not None else np.arange(n,len(ids))
                    at=x[t]
                    def objective(w):
                        z=native[t]+at@w
                        return np.logaddexp(0,z).sum()-y[t]@z+16*(w@w),at.T@(expit(z)-y[t])+32*w
                    res=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
                    assert res.success
                    z=native[q]+x[q]@res.x
                    if j is None:pack.append(z)
                    else:value[q]=z
            value[n:]=np.mean(pack,axis=0)
            bk=scheme+'_core'
            if label=='control':
                outputs[bk]=value
                continue
            ctrl=outputs[bk]
            inc=(value-ctrl)*np.where(first,-1.,1.)
            actual=(value>=0)!=first
            novel=actual&((ctrl>=0)==first)
            centered=(inc>abs(native))&actual
            for region,m in [('retained',np.ones(len(ids),bool)),('split',(nv>0)&(nv<3)),('agree',nv==0)]:
                for tag,flip in [('direct',actual),('novel',novel),('anchored',centered)]:
                    v=flip&~frozen&m
                    key='_'.join([label,region,tag,scheme])
                    outputs[key]=v
                    delta=v[:n]*np.where(parent[:n]!=y,1,-1)
                    report[key]=dict(test=int(v[n:].sum()),fold_net=[int(delta[known&(f==k)].sum()) for k in range(8)],
                        scopes={s:count(v[:n],m,parent[:n]!=y) for s,m in [('all',known),('fresh',fresh&known),('common31',blind)]})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',ids=ids,y=y,f=f,parent=parent,dpt=dpt,tabnet=net,**outputs)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():
        a=v['scopes']['all'];fr=v['scopes']['fresh']
        if a['fixed']>a['harmed']:print(k,a,fr,v['test'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['test_dpt','report'])
    globals()[p.parse_args().task]()
