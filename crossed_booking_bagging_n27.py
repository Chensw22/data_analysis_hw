"""Compare actual delete-fold bags with full-size fits on unseen bookings and the competition manifest.

On the project Mac run from the repository root:
DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs OMP_NUM_THREADS=1 .venv/bin/python crossed_booking_bagging_n27.py
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from directional_foundation_transfer_n27 import calibrate
from full_roster_booking_validation_n23 import ROOT, data
from price_implied_deck_n26 import build_features
from repeated_archive_misses_n26 import ART, cohort, count
from six_way_against_scored_nine_n26 import test_decisions

OUT=ART/'nov27_crossed_booking_bagging'
REPORT=ROOT/'reports/nov23_matched_learning_curve/crossed_booking_bagging.json'
SEEDS=[3027,4073,5197,6311,7433,8537,9649,10753]


def fit(x,y,t,q,seed):
    model=lgb.LGBMClassifier(n_estimators=270,learning_rate=.035,num_leaves=21,min_child_samples=35,
             reg_lambda=4,force_col_wise=True,n_jobs=1,verbosity=-1,random_state=seed,
             subsample=.9,subsample_freq=1,colsample_bytree=.9)
    return model.fit(x[t],y[t]).predict_proba(x[q])[:,1]


def load_fit(label,x,y,g,ids,t,q,seed):
    path=OUT/(label+'.npz')
    assert not np.isin(g[t],g[q]).any()
    if not path.exists():
        p=fit(x,y,t,q,seed)
        temp=path.with_suffix('.tmp.npz')
        np.savez_compressed(temp,t=t,q=q,labels=y[t],pid=ids[q],p=p,seed=seed)
        temp.replace(path)
    with np.load(path) as z:
        for k,v in [('t',t),('q',q),('labels',y[t]),('pid',ids[q]),('seed',seed)]:
            np.testing.assert_array_equal(z[k],v)
        return z['p']


def decisions(ensemble,single,parent,foundation,tag):
    side=np.where(parent[:,None],1-ensemble,ensemble)
    risk=side.min(axis=1)  # The least confident of eight must still predict the opposite outcome.
    if tag=='new_consensus':risk=np.where((single>=.5)==parent,risk,0.)
    if tag=='independent_support':
        other=np.where(parent[:,None],1-foundation,foundation)
        risk=np.minimum(risk,np.median(other,axis=1))
    if tag=='single_control': risk=np.where(parent,1-single,single)
    return risk


def calibrations(score,base,y,g,fold,known,ordered=False):
    n=len(y)
    mask=np.zeros(n,bool)
    cuts=[]
    for k in range(8):
        group=np.flatnonzero(fold==k)
        bookings=np.unique(g[group])
        if not ordered:bookings=np.random.default_rng(5417+k).permutation(bookings)
        for part in np.array_split(bookings,4):
            q=group[np.isin(g[group],part)]
            t=group[known[group]&~np.isin(g[group],part)]
            assert not np.isin(g[t],g[q]).any()
            cut=calibrate(score,base,y,t,1.)
            mask[q]=score[q]>=cut[base[q].astype(int)]
        cuts.append(calibrate(score,base,y,group[known[group]],1.))
    cuts=np.asarray(cuts)
    tst=(score[n:,None]>=cuts[:,base[n:].astype(int)].T).sum(axis=1)>4
    return np.r_[mask,tst],cuts


def run():
    raw,y,g,random=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    roster=pd.concat([raw,test],ignore_index=True)
    groups=roster.PassengerId.str.split('_').str[0].to_numpy()
    ids=roster.PassengerId.to_numpy(str)
    _,blind,known,fresh=cohort(raw,y,random)
    with np.load(ART/'nov27_manifest_group_dro/aligned.npz') as z:
        np.testing.assert_array_equal(ids,z['passenger']);np.testing.assert_array_equal(random,z['fold'])
        ordered,base=z['ordered'],z['anchor']
    parent_version='proposal_n27_directional_foundation'
    official,parent_sha=test_decisions(parent_version,test.PassengerId)
    np.testing.assert_array_equal(official,base[n:])
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(ids,z['passenger'])
        foundation=z['scores'][:,-3:]
    x=pd.get_dummies(build_features(roster,groups)['fine'],dtype=float).to_numpy(dtype=float)
    OUT.mkdir(exist_ok=True)
    folds={'random':random,'ordered':ordered}
    test_ids=np.arange(n,len(ids))
    bags={m:np.full((len(ids),8),np.nan) for m in folds}
    single={m:np.empty(len(ids)) for m in folds}
    # Each crossed training set excludes both a random whole-order region and a contiguous one.
    # Its scores are permissible in both outer protocols. A Latin square gives each bag eight unique seeds.
    for a in range(8):
        for b in range(8):
            q=np.flatnonzero((random==a)|(ordered==b))
            t=np.flatnonzero((random!=a)&(ordered!=b))
            s=SEEDS[(a+b)%8]
            p=load_fit(f'cell_{a}_{b}',x,y,groups,ids,t,q,s)
            ma=random[q]==a;mb=ordered[q]==b
            bags['random'][q[ma],b]=p[ma];bags['ordered'][q[mb],a]=p[mb]
        print('crossed',a,flush=True)
    seed_control=np.column_stack([load_fit(f'full_{j}',x,y,groups,ids,np.arange(n),test_ids,s) for j,s in enumerate(SEEDS)])
    single_test=seed_control[:,0]
    for mode,fold in folds.items():
        # Random-region holdouts use inner contiguous deletions in validation, and vice versa.
        target='ordered' if mode=='random' else 'random'
        for a in range(8):
            t=np.flatnonzero(fold!=a);h=np.flatnonzero(fold==a);q=np.r_[h,test_ids]
            native=load_fit(f'{mode}_{a}_fixed',x,y,groups,ids,t,q,SEEDS[0])
            varied=load_fit(f'{mode}_{a}_varied',x,y,groups,ids,t,q,SEEDS[a]) if a else native
            single[mode][h]=native[:len(h)]
            bags[target][n:,a]=varied[len(h):]
        single[mode][n:]=single_test
    masks={'all':known,'fresh':known&fresh,'common':blind}
    report={'parent':parent_version,'parent_sha256':parent_sha,'full_seed_variation':dict(
        changed_vs_single=int(((seed_control.mean(1)>=.5)!=(single_test>=.5)).sum()),
        mean_passenger_sd=float(seed_control.std(axis=1).mean()))}
    save={}
    for mode,fold in folds.items():
        bag=bags[mode];raw_score=single[mode]
        assert np.isfinite(bag).all()
        avg=bag.mean(1)
        report[mode]={'scores':{},'rules':{}}
        for tag,idx in [('oof',np.arange(n)),('test',test_ids)]:
            sides=bag[idx]>=.5
            mean=avg[idx]>=.5
            ref=raw_score[idx]>=.5
            data_row=dict(rows=len(idx),mean_absolute_probability_delta=float(abs(avg[idx]-raw_score[idx]).mean()),
                          mean_per_person_bag_sd=float(bag[idx].std(1).mean()),threshold_crossings=int((mean!=ref).sum()),
                          all_eight_agree=int((sides==sides[:,:1]).all(1).sum()),
                          unanimous_contradictions_of_single=int((sides!=ref[:,None]).all(1).sum()))
            if tag=='oof':data_row.update(full_correct=int((ref==y).sum()),bag_correct=int((mean==y).sum()))
            else:data_row.update(mean_absolute_delta_vs_full_seed_bag=float(abs(avg[idx]-seed_control.mean(1)).mean()),
                                 flips_vs_full_seed_bag=int((mean!=(seed_control.mean(1)>=.5)).sum()))
            report[mode]['scores'][tag]=data_row
        for tag in ['single_control','bag','independent_support','new_consensus']:
            score=decisions(bag,raw_score,base,foundation,tag)
            save[mode+'_'+tag+'_risk']=score
            actions={}
            for cutmode in ['random','ordered']:
                flag,cuts=calibrations(score,base,y,g,fold,known,ordered=cutmode=='ordered')
                actions[cutmode]=flag
                save[mode+'_'+tag+'_cuts']=cuts
            np.testing.assert_array_equal(actions['random'][n:],actions['ordered'][n:])
            for cutmode,flag in [*actions.items(),('joint',actions['random']&actions['ordered'])]:
                d=flag[:n]*np.where(base[:n]!=y,1,-1)
                report[mode]['rules'][tag+'_'+cutmode]=dict(test_changes=int(flag[n:].sum()),
                    folds=[int(d[known&(fold==k)].sum()) for k in range(8)],
                    scopes={s:count(flag[:n],m,base[:n]!=y) for s,m in masks.items()})
                save[mode+'_'+tag+'_'+cutmode]=flag
        # Transfer the measured training-size log-odds change rather than using absolute tree confidence.
        # Require at least six deletion fits to favor the competing outcome.
        foundation_margin=np.abs(logit(foundation.mean(1).clip(.001,.999)))
        opposite_shift=np.where(base,-1,1)*(logit(avg)-logit(raw_score))
        quorum=((bag>=.5)!=base[:,None]).sum(1)
        for votes in [6,8]:
            flag=(opposite_shift>foundation_margin)&(quorum>=votes)
            d=flag[:n]*np.where(base[:n]!=y,1,-1)
            key=f'logodds_increment_q{votes}'
            report[mode]['rules'][key]=dict(test_changes=int(flag[n:].sum()),
                folds=[int(d[known&(fold==k)].sum()) for k in range(8)],
                scopes={s:count(flag[:n],m,base[:n]!=y) for s,m in masks.items()})
            save[mode+'_'+key+'_joint']=flag
        save[mode+'_bag_scores']=bag;save[mode+'_single']=raw_score
        print(mode,report[mode]['scores'],[(k,v['scopes']['all'],v['test_changes']) for k,v in report[mode]['rules'].items() if k.endswith('_joint')],flush=True)
    cross={}
    for tag in ['single_control','bag','independent_support','new_consensus','logodds_increment_q6','logodds_increment_q8']:
        flag=save['random_'+tag+'_joint']&save['ordered_'+tag+'_joint']
        cross[tag]=dict(test_changes=int(flag[n:].sum()),scopes={s:count(flag[:n],m,base[:n]!=y) for s,m in masks.items()})
        save['cross_'+tag]=flag
    report['cross_protocol_same_passenger']=cross
    report['notes']=['Each of eight nested models omits the complete outer booking fold and its own inner deletion fold. Crossed random/contiguous folds share fits; their results overlap.',
          'The nested validation reference uses seven eighths of the training data and its bags use roughly 49/64; the test reference uses all training orders and its bags seven eighths.',
          'All decision thresholds are calibrated on different bookings in the same external fold. Two independent calibration partitions assess threshold fragility.',
          'The same cross-fitted champion replay is used as reference for both validations; the new fare models are independently refit per protocol. The deployed ensembles are real predictions from training-sized leave-fold models.']
    np.savez_compressed(OUT/'aligned.npz',ids=ids,anchor=base,y=y,ordered=ordered,random=random,full_seeds=seed_control,**save)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(cross,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
