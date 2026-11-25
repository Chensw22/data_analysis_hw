"""Repartition the five fixed-bill models and evaluate their preselected quarter mixture."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from certified_tree_release_n25 import AUDIT, BEST, HISTORY, NAME as PREVIOUS, structural_inputs
from export_budget_quarter_n23 import source
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,data
from full_roster_fixed_bill_margins_n24 import matrices,MODES
from full_roster_permutation_selection_n24 import cat
from full_roster_solo_coverage_n24 import cached
from kaggle_workflow import sha256,validate_submission
from production_anchor_cpu_replay_n25 import certified
from relational_error_corrector import write


SEEDS=(251141,251143)
NAME='proposal_n25_certified_bill_margins'
REPORT=ROOT/'reports/nov23_matched_learning_curve/certified_bill_margins.json'
CACHE=ROOT/'artifacts/nov25_certified_bill_margins'
OUT=ROOT/'artifacts'/NAME


def features(train,test):
    # Compute features on each complete roster separately, as in the archived folds.
    # Confirm the categorical expansion has precisely the same layout.
    def schema(raw):
        cabin=raw.Cabin.str.split('/',expand=True)
        reg=raw[['HomePlanet','Destination','VIP']].assign(Deck=cabin[0],Side=cabin[2])
        return [list(pd.get_dummies(reg[[c]].astype('category'),dummy_na=True).columns) for c in reg]
    assert schema(train)==schema(test)
    inputs=[]
    for part,frame in [('train',train),('test',test)]:
        with np.load(ROOT/f'artifacts/nov24_bill_allocation_cycles/{part}.npz') as z:
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
            values=matrices(frame,frame.PassengerId.str.split('_').str[0].to_numpy(),z['counter'])
        inputs.append(values)
    return inputs


def forecast(models,x,ids):
    scores=[m.predict_proba(v[ids],thread_count=3)[:,1] for m,v in zip(models,x,strict=True)]
    scores.append(models[2].predict_proba(x[3][ids],thread_count=3)[:,1])
    return np.column_stack(scores)


def folds(g,seed):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}/{v}'.encode()).digest()[:8],'big')%8 for v in g])


def fit_train(xs,y,g,seed):
    p=np.full((len(y),len(MODES)),np.nan)
    models=[]
    for j,(t,q) in enumerate(GroupKFold(8,shuffle=True,random_state=seed).split(y,groups=g)):
        assert not np.isin(g[t],g[q]).any()
        cache=CACHE/f'{seed}_{j}.joblib'
        if cache.exists():
            record=joblib.load(cache)
            for k,v in [('t',t),('q',q),('labels',y[t])]: np.testing.assert_array_equal(record[k],v)
            pack=record['models']
        else:
            pack=[cat(497).fit(a[t],y[t]) for a in xs]
            joblib.dump(dict(t=t,q=q,labels=y[t],models=pack),cache)
        p[q]=forecast(pack,xs,q)
        models.append(pack)
    assert np.isfinite(p).all()
    return p,models


def test_on_models(models,xt,g):
    banks=[]
    for seed,parts in zip(SEEDS,models,strict=True):
        fold=folds(g,seed)
        p=np.full((len(g),len(MODES)),np.nan)
        for j,pack in enumerate(parts):
            ids=np.flatnonzero(fold==j)
            p[ids]=forecast(pack,xt,ids)
        assert np.isfinite(p).all()
        banks.append(p)
    return np.stack(banks)


def edits(a,rival,inc,mask):
    s=.75*a+.25*rival
    return mask&(inc==(a>=.5))&((s>=.5)!=inc),s


def tally(e,right,mask):
    return dict(fixed=int((e&~right&mask).sum()),harmed=int((e&right&mask).sum()))


def run():
    raw,y,g,f=data()
    _,oy,og,rows,of,_,obase,_,_,_,_,old,_=aligned()
    testfile=ROOT/'data/raw/test.csv'
    test=pd.read_csv(testfile,usecols=lambda c:c!='Name')
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    assert not np.isin(tg,g).any()
    fields=structural_inputs(pd.concat([raw,test],ignore_index=True))
    hc=certified(old[:,0],*[a[rows] for a in fields])
    with np.load(AUDIT/'oof.npz') as z:
        for k,v in [('row',np.arange(len(y))),('y',y),('fold',f)]:np.testing.assert_array_equal(z[k],v)
        cert,a,inc=z['certified_v131_equals_v96'],z['v96'],z['certified_nine'].astype(bool)
    with np.load(HISTORY/'historical.npz') as z:
        for k,v in [('rows',rows),('v131',obase),('y',oy)]: np.testing.assert_array_equal(z[k],v)
        hist=z['nine']
    with np.load(AUDIT/'certified_test.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        tq=z['certified_v131_equals_v96']
    a_test,_,ah=source('proposal_online_first_counts15',test)
    soft,actual,bh=source(BEST,test)
    _,previous,ph=source(PREVIOUS,test)
    pre_edits=actual!=previous
    assert (cert.sum(),hc.sum(),tq.sum(),pre_edits.sum())==(4111,3300,1989,67)
    xs,xt=features(raw,test)
    CACHE.mkdir(exist_ok=True)
    tables=[]; models=[]
    for seed in SEEDS:
        scores,pack=fit_train(xs,y,g,seed)
        tables.append(scores); models.append(pack)
        print('complete',seed,flush=True)
    testbanks=test_on_models(models,xt,tg)
    for u,c in [(tables,cert),(testbanks,tq)]:
        for scores in u: np.testing.assert_array_equal(scores[c,2],scores[c,4])
    train_rivals=[np.median(p,axis=1) for p in tables]
    test_rivals=[np.median(p,axis=1) for p in testbanks]
    new=~np.isin(g,og)
    report={}
    for k,p,pt in zip([*map(str,SEEDS),'two_fit_mean'],[*train_rivals,np.mean(train_rivals,axis=0)],
                      [*test_rivals,np.mean(test_rivals,axis=0)],strict=True):
        e,_=edits(a,p,inc,cert)
        h,_=edits(old[:,0],p[rows],hist,hc)
        te,ps=edits(a_test,pt,actual,tq)
        same=e[rows]&h
        report[k]=dict(eight=tally(e,inc==y,cert),new_orders=tally(e,inc==y,cert&new),
            eight_fold_net=[int(((inc!=y)&e&(f==v)).sum()-((inc==y)&e&(f==v)).sum()) for v in np.unique(f)],
            historical=tally(h,hist==oy,hc),common=dict(fixed=int((same&(inc[rows]!=oy)&(hist!=oy)).sum()),
            harmed=int((same&(inc[rows]==oy)&(hist==oy)).sum())),test_edits=int(te.sum()),
            overlap_previous=int((te&pre_edits).sum()))
    edit_a,_=edits(a,train_rivals[0],inc,cert)
    edit_b,_=edits(a,train_rivals[1],inc,cert)
    report['two_seed_common']=dict(eight=tally(edit_a&edit_b,inc==y,cert),
        new_orders=tally(edit_a&edit_b,inc==y,cert&new))
    report['same_fit_measured_equals_same_fit_reconstructed']=True
    report['archive_replay']={}
    masks=np.unique(g,return_inverse=True,return_counts=True)
    assert np.all(masks[2][masks[1]][cert]==1)
    for tag,ix,fold,mask,anchor,baseline in [('full',np.arange(len(y)),f,cert,a,inc),
                                             ('historical',rows,of,hc,old[:,0],hist)]:
        score=cached('fixed_bill_margins',tag,ix,fold,g,y)
        mean=np.median(score,axis=1)
        move,_=edits(anchor,mean,baseline,mask)
        entry=tally(move,baseline==y[ix],mask)
        entry.update(rows=int(mask.sum()),
                     same_fit_identical=int((score[mask,2]==score[mask,4]).sum()),
                     separately_trained_identical=int((score[mask,2]==score[mask,3]).sum()),
                     median_equals_measured=int((mean[mask]==score[mask,2]).sum()))
        assert (entry['fixed'],entry['harmed'])==((37,22) if tag=='full' else (35,21))
        report['archive_replay'][tag]=entry
    write(REPORT,report)
    print(report,flush=True)
    if not all(report[str(seed)]['eight']['fixed']>report[str(seed)]['eight']['harmed'] for seed in SEEDS):
        return
    OUT.mkdir(exist_ok=True)
    model=OUT/'models.joblib'
    joblib.dump(models,model)
    reloaded=np.median(test_on_models(joblib.load(model),xt,tg),axis=2).mean(0)
    np.testing.assert_array_equal(pt,reloaded)
    mask,s=edits(a_test,pt,actual,tq)
    replay_mask,rs=edits(a_test,reloaded,actual,tq)
    pred=np.where(mask,s,soft); rep=np.where(replay_mask,rs,soft)
    np.testing.assert_array_equal(pred,rep)
    np.testing.assert_array_equal((pred>=.5)!=actual,mask)
    pd.DataFrame(dict(PassengerId=test.PassengerId,probability=pred,reload_probability=rep)).to_csv(OUT/'test_probabilities.csv',index=False)
    csv=OUT/'submission.csv'
    pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=pred>=.5)).to_csv(csv,index=False)
    check=validate_submission(csv,testfile)
    digests=[json.loads(p.read_text()).get('file_sha256') for p in (ROOT/'reports').glob('*/online_attempts/*.json')]
    assert check['sha256'] not in digests
    version=NAME.removeprefix('proposal_')
    manifest=dict(**check,version=version,model_sha256=sha256(model),reload_difference=0,training_rows=len(y),
                  parents={BEST:bh,'v96':ah,'comparison_certified_tree':ph},edits_vs_best=int(mask.sum()))
    specification=dict(version=version,seeds=list(SEEDS),validation='eight held-out whole bookings folds per seed',
        arms=list(MODES),rule='Average per-seed medians, weight 25% versus 75% original Mitra; preserve nine and require certification.',
        evidence=str(REPORT.relative_to(ROOT)))
    for folder in (OUT,ROOT/'reports'/NAME):
        write(folder/'manifest.json',manifest);write(folder/'selection.json',specification)


if __name__=='__main__':
    with threadpool_limits(2):run()
