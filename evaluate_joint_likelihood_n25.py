"""Audit new generative and missing-source decisions on the complete out-of-fold error population."""
import argparse
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold

from disjoint_tree_fields_n25 import ART,DEST,stats
from evaluate_awake_demonstrations_n25 import deployed
from evaluate_cross_architecture_n25 import offset_fit
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF

OUT=ART/'nov25_passenger_joint_likelihood'


def odds(p):
    return logit(np.clip(p,.002,.998))


def proposals(z,groups,y,native,k):
    sources=[name[5:] for name in z.files if name.startswith('omit_')]
    omit=np.column_stack([z['omit_'+name] for name in sources])
    restore=np.column_stack([z['restore_'+name] for name in sources])
    direct=z['direct']
    order=np.column_stack([z['chain_0'],z['chain_1']])
    gen=expit(odds(order).mean(1))
    reconstructed=expit(odds(restore).mean(1))
    # When learning Y from the whole manifest and reconstructing one source under
    # each candidate Y disagree, the two heads imply incompatible likelihood ratios.
    incoherence=np.abs(odds(restore)-odds(direct)[:,None])
    selected=np.argmax(incoherence,axis=1)
    shuffled=np.random.default_rng(185+k).permutation(selected)
    ix=np.arange(len(y))
    guarded=np.where((omit[ix,selected]>=.5)==(restore[ix,selected]>=.5),omit[ix,selected],direct)
    result={'outcome_only':z['outcome_only'],'joint_direct':direct,'registration_first':order[:,0],
            'bills_first':order[:,1],'mean_chains':gen,'restore_mean':reconstructed,'guarded_omission':guarded,
            'most_inconsistent_omitted':omit[ix,selected],'random_source_omitted':omit[ix,shuffled]}
    for j,name in enumerate(sources):
        result['omit_'+name]=omit[:,j]
        result['restore_'+name]=restore[:,j]
    natural=odds(native)
    common=np.column_stack([np.ones(len(y)),natural])
    base=odds(direct)
    bank=np.column_stack([base-natural,odds(order)-natural[:,None],odds(omit)-base[:,None],odds(restore)-base[:,None]])
    rng=np.random.default_rng(602+k)
    bins=np.minimum((np.clip(native,0,1)*10).astype(int),9)
    placebo=bank.copy()
    for v in np.unique(bins):
        idx=np.flatnonzero(bins==v)
        placebo[idx]=bank[rng.permutation(idx)]
    for name,features in [('native_calibrated',common),('learned_sources',np.column_stack([common,bank])),
                          ('placebo_sources',np.column_stack([common,placebo]))]:
        p=np.full(len(y),np.nan)
        for t,q in GroupKFold(4,shuffle=True,random_state=519+k).split(y,groups=groups):
            assert not np.isin(groups[t],groups[q]).any()
            p[q]=offset_fit(features,natural,y,t,q)
        assert np.isfinite(p).all()
        result[name]=p
    return result


def run(folds):
    raw,y,g,fold=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,fold,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    majority=votes>34
    all_agree=(votes==0)|(votes==68)
    near=(votes<=3)|(votes>=65)
    results={}
    indices=[]
    coverage=[]
    for k in folds:
        q=np.flatnonzero(fold==k)
        with np.load(REF/f'release{k}.npz') as anchor:
            np.testing.assert_array_equal(anchor['q'],q)
            p,prior,lo,hi=(anchor[key] for key in ('native','v131','nine_on_low_branch','nine_on_high_branch'))
        known=prior>=0
        indices.append((q,known))
        with np.load(OUT/f'fold{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['train_y'],y[z['t']])
            candidates=proposals(z,g[q],y[q],p,k)
        native=p>=.5
        actual=candidates.copy()
        for name in ('joint_direct','mean_chains','restore_mean','guarded_omission'):
            actual[name+'_quarter']=.75*p+.25*candidates[name]
        raw_modes=['outcome_only','joint_direct','registration_first','bills_first',
                   *[n for n in candidates if n.startswith(('omit_','restore_')) and n!='restore_mean']]
        hits=(np.column_stack([candidates[n] for n in raw_modes])>=.5)==y[q,None]
        coverage.append((hits.any(1),(~hits).any(1)))
        for name,score in actual.items():
            sign=score>=.5
            sides=[]
            for s,nine in [(False,lo),(True,hi)]:
                dep=deployed(sign,np.where(known,prior,s),nine,native)
                sides.append((dep==y[q]).astype(int)-(nine==y[q]))
            archive=(sign==y[q]).astype(int)-(majority[q]==y[q])
            results.setdefault(name,[]).append((*sides,archive))
    ix=np.concatenate([q for q,_ in indices])
    known=np.concatenate([m for _,m in indices])
    all_same=all_agree[ix]
    wrong=majority[ix]!=y[ix]
    some_correct=np.concatenate([c for c,_ in coverage])
    some_incorrect=np.concatenate([c for _,c in coverage])
    reach={}
    for name,mask in [('unanimous',all_same),('near_unanimous',near[ix]),('whole',np.ones(len(ix),bool))]:
        reach[name]=dict(rows=int(mask.sum()),errors=int((mask&wrong).sum()),
            at_least_one_correct_alternative=int((mask&wrong&some_correct).sum()),
            original_correct_exposed=int((mask&~wrong&some_incorrect).sum()))
    records={}
    for name,blocks in results.items():
        a,b,d=(np.concatenate([v[j] for v in blocks]) for j in range(3))
        np.testing.assert_array_equal(a[known],b[known])
        def calc(delta,mask):return stats(delta,mask,fold[ix],g[ix])
        records[name]=dict(bounds=[int(np.minimum(a,b).sum()),int(np.maximum(a,b).sum())],
            known=calc(a,known),unseen=calc(a,known&~np.isin(g[ix],g[rows])),
            consensus=calc(d,all_same),near=calc(d,near[ix]))
    result=dict(folds=folds,rows=len(ix),known=int(known.sum()),opportunity=reach,rules=records)
    path=DEST.with_name('joint_manifest_likelihood.json') if len(folds)==8 else OUT/'interim.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('Opportunity:',reach,flush=True)
    for n,r in records.items():
        print(n,r['bounds'],r['known']['fixed'],r['known']['harmed'],r['consensus']['fixed'],r['consensus']['harmed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',type=int,choices=range(8),default=list(range(8)))
    run(parser.parse_args().folds)
