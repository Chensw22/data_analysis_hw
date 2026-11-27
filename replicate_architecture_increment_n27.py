"""Repeat the pre-existing 45-person classifier on fresh order partitions and compare both scored parents."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from repeated_archive_misses_n26 import ART,cohort,count
from export_consensus_review_n26 import ROOT,write,dig
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from six_way_against_scored_nine_n26 import test_decisions

BASE='proposal_n27_directional_foundation'
NAME='proposal_n27_architecture_increment'
REPORT=ROOT/'reports/nov23_matched_learning_curve/replicate_architecture_increment.json'


def predict(x,prior,t,q,y):
    a=x[t]
    def objective(w):
        z=prior[t]+a@w
        return np.logaddexp(0,z).sum()-y[t]@z+16*(w@w), a.T@(expit(z)-y[t])+32*w
    res=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    assert res.success
    return prior[q]+x[q]@res.x


def run(export):
    raw,y,g,f=data()
    n=len(y)
    _,blind,known,fresh=cohort(raw,y,f)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        foundations=z['scores'][:,-3:]
        ids=z['passenger']
        np.testing.assert_array_equal(z['y'],y)
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        first=z['first']; official=z['random_median']
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        net,dpt=z['tabnet'],z['dpt']
        target=z['spread_split_anchored_random'][n:]
    lg=lambda p:logit(p.clip(.005,.995))
    med=np.median(foundations,axis=1)
    risk=np.where(first,1-med,med)
    native=lg(med)
    core=np.column_stack([np.ones(len(ids)),lg(foundations)])
    full=np.column_stack([core,lg(dpt),lg(net.mean(1)),net.std(1),(foundations>=.5).mean(1)])
    splitvote=(np.where(first[:,None],1-foundations,foundations)>.5).sum(1)
    divided=(splitvote>0)&(splitvote<3)
    test=np.arange(n,len(ids))
    scores=[]
    for k in range(8):
        t=np.flatnonzero(f==k)
        scores.append(np.column_stack([predict(x,native,t,test,y) for x in [core,full]]))
    test_scores=np.mean(scores,axis=0)
    def activation(sc,base):
        inc=(sc[:,1]-sc[:,0])*np.where(base,-1.,1.)
        return ((sc[:,1]>=0)!=base)&(inc>abs(sc[:,2]))
    actual=activation(np.column_stack([test_scores,native[n:]]),first[n:])&divided[n:]&~official[n:]
    np.testing.assert_array_equal(target,actual)
    reps=[]
    summary={}
    masks={}
    for name,side in [('random',splits(g,f,'random')),('ordered',splits(g,f,'contiguous'))]+[(str(i),partition(g,'increment_replica_'+str(i),4)) for i in range(8)]:
        cv=np.empty((n,2))
        lock=np.empty(n,bool)
        for k in range(8):
            for j in range(4):
                t=np.flatnonzero((f==k)&(side!=j));q=np.flatnonzero((f==k)&(side==j))
                assert not np.isin(g[t],g[q]).any()
                lock[q]=risk[q]>=calibrate(risk,first,y,t[known[t]],1.)[first[q].astype(int)]
                for col,x in enumerate((core,full)):cv[q,col]=predict(x,native,t,q,y)
        flag=activation(np.column_stack([cv,native[:n]]),first[:n])&divided[:n]&~lock
        err=first[:n]!=y
        summary[name]=dict(all=count(flag,known,err),fresh=count(flag,known&fresh,err),old=count(flag,blind,err))
        if name in ('ordered','random'):
            masks[name]=np.r_[flag,actual]
            masks[name+'_parent']=first^np.r_[lock,official[n:]]
        else:reps.append(flag)
    flag=np.mean(reps,axis=0)>.5
    summary['majority']={s:count(flag,m,err) for s,m in [('all',known),('fresh',fresh&known),('old',blind)]}
    with np.load(ART/'nov27_payment_state_routes/aligned.npz') as z:
        route=z['random'][n:].any(1)
    report=dict(splits=summary,changed=int(actual.sum()),overlap_payment_states=int((actual&route).sum()),
                versus_payment_states=int((actual!=route).sum()))
    cache=ART/'nov27_repeated_architecture_increment'
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache/'aligned.npz',passenger=ids,y=y,fold=f,**masks)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)
    if not export:return
    hard,sha=test_decisions(BASE,pd.Series(ids[n:]))
    par=pd.read_csv(ART/BASE/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(par.PassengerId,ids[n:])
    np.testing.assert_array_equal(par.probability>=.5,hard)
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    np.savez_compressed(model,passenger=ids[n:],parent=par.probability.to_numpy(),base=first[n:],native=native[n:],
                        scores=test_scores,divided=divided[n:],preserve=official[n:])
    def apply(parent,base,native,scores,divided,preserve):
        use=activation(np.column_stack([scores,native]),base)&divided&~preserve
        return np.where(use,expit(scores[:,1]),parent)
    p=apply(par.probability,first[n:],native[n:],test_scores,divided[n:],official[n:])
    with np.load(model) as z:replay=apply(*(z[k] for k in ('parent','base','native','scores','divided','preserve')))
    np.testing.assert_array_equal(p,replay)
    np.testing.assert_array_equal(p>=.5,hard^actual)
    csv=dest/'submission.csv'
    pd.DataFrame({'PassengerId':ids[n:],'Transported':p>=.5}).to_csv(csv,index=False)
    pd.DataFrame({'PassengerId':ids[n:],'probability':p,'reload_probability':replay}).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(path.read_text())['file_sha256']!=check['sha256'] for path in (ROOT/'reports').glob('*/online_attempts/*.json'))
    entry=dict(**check,version=NAME.removeprefix('proposal_'),parents={BASE:sha},training_rows=n,model_sha256=dig(model),
               reload_difference=0,changed=int(actual.sum()),source_file='replicate_architecture_increment_n27.py')
    for folder in (dest, ROOT/'reports'/NAME):
        write(folder/'manifest.json',entry)
        write(folder/'selection.json',dict(version=entry['version'],rule='spread_split_anchored',evidence=str(REPORT.relative_to(ROOT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    run(parser.parse_args().export)
