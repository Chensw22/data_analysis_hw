"""Measure held-order challenges after the 161 release, Nori adjacency and difficulty all retain a passenger.

Use .venv/bin/python. Calibration uses only historical bookings held out by the
original fold models, excluding the inner query orders. Newly added model
information is represented by within-architecture differences in scores.
"""
import json
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_roster_booking_validation_n23 import ROOT, data
from nori_booking_regression_n26 import FEES
from rtj_checkpoint_setup_n26 import WEIGHTS

ART = ROOT/'artifacts'
LATENT = ART/'nov26_latent_passenger_difficulty'
OUT = ART/'nov26_retained_order_mechanisms'
REPORT = ROOT/'reports/nov23_matched_learning_curve/retained_order_mechanisms.json'


def check(z, expected):
    for key, value in expected.items():
        np.testing.assert_array_equal(z[key], value)


def paired_scores(raw, y, g, f):
    ids = raw.PassengerId.to_numpy(str)
    aligned = dict(passenger=ids, row=np.arange(len(y)), y=y, fold=f)
    pairs = {}
    for name, path, before, after in [
        ('Nori_companion', 'nov26_nori_fare_party/aligned.npz', 'own', 'peer'),
        ('Boolean_conjunctions', 'nov26_sparse_boolean_clauses/eight_fold_oof.npz', 'additive', 'full'),
        ('Soft_matching_capacity', 'nov26_relaxed_passenger_transport/eight_fold_oof.npz', 'local', 'relaxed')]:
        with np.load(ART/path) as z:
            check(z, aligned)
            pairs[name] = np.column_stack([z[before],z[after]])
    caps, graph = [np.full((len(y),2),np.nan) for _ in range(2)]
    for k in np.unique(f):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        runs=[]
        for seed in [4537,8912]:
            with np.load(ART/f'nov26_booking_capsules/fold{k}_{seed}.npz') as z:
                check(z,dict(q=q,t=t,labels=y[t],ids=ids[q]))
                runs.append(np.column_stack([z['fixed'],z['dynamic']]))
        caps[q] = np.mean(runs,axis=0)
        for j,variant in enumerate(['no_edges','edges']):
            with np.load(ART/f'nov26_rtj_probe/oof/{k}_{variant}.npz') as z:
                check(z,dict(ids=ids[q],y=y[q]))
                assert str(z['weights'])==WEIGHTS and str(z['context_policy'])=='mask_foreign_query_task_nodes_v1'
                graph[q,j]=expit(z['logits'])
    pairs['Capsule_routing']=caps
    pairs['RTJ_booking_edges']=graph
    x=np.stack(list(pairs.values()),axis=1)
    assert np.isfinite(x).all()
    return list(pairs), x


def counteracts(scores,anchor):
    return (scores>=.5)!=anchor.reshape((-1,)+(1,)*(scores.ndim-1))


def tally(action, wrong, mask, f):
    use=action&mask
    delta=np.where(use,np.where(wrong,1,-1),0)
    return dict(fixed=int((use&wrong).sum()),harmed=int((use&~wrong).sum()),net=int(delta.sum()),
                fold_net=np.bincount(f,weights=delta).astype(int).tolist())


def calibrate(x, controls, median, wrong, groups, f, history, ordered):
    # Use a reference that is purely a function of the frozen first-level scores.
    # Do not train on eligibility defined by another calibration patient's label.
    sign=np.where(median,-1.,1.)[:,None]
    old=sign*logit(x.clip(.002,.998))
    paired=sign[:,:,None]*logit(controls.clip(.002,.998))
    quantities=np.column_stack([median,old.mean(1),old.max(1),old.std(1),np.mean(old>0,axis=1)])
    before=paired[:,:,0]
    base=np.column_stack([quantities,before])
    gain=paired[:,:,1]-before
    designs=(base,np.column_stack([base,gain]))
    estimates=np.full((len(f),len(designs)),np.nan)
    for k in range(8):
        ids=np.flatnonzero(f==k)
        if ordered:
            buckets=np.array_split(np.unique(groups[ids]),4)
            splits=[(ids[~np.isin(groups[ids],b)],ids[np.isin(groups[ids],b)]) for b in buckets]
        else:
            splits=[(ids[a],ids[b]) for a,b in GroupKFold(4,shuffle=True,random_state=8123+k).split(ids,groups=groups[ids])]
        for t,q in splits:
            t=t[history[t]]
            assert len(t)>0 and not np.isin(groups[t],groups[q]).any()
            for j,design in enumerate(designs):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=400))
                model.fit(design[t],wrong[t])
                estimates[q,j]=model.predict_proba(design[q])[:,1]
    assert np.isfinite(estimates).all()
    return estimates


def run():
    raw,y,g,f=data()
    n=len(y)
    ids=raw.PassengerId.to_numpy(str)
    with np.load(LATENT/'panel.npz') as z:
        check(z,dict(y=y,fold=f))
        np.testing.assert_array_equal(z['passenger'][:n],ids)
        np.testing.assert_array_equal(z['groups'][:n],g)
        base,known,common,fresh,x,names=[z[k][:n] for k in ['anchor','known','common','fresh','scores','names']]
    models,v=paired_scores(raw,y,g,f)
    with np.load(ART/'nov26_nori_adjacent_manifest/aligned.npz') as z:
        check(z,dict(y=y,fold=f,passenger=ids,reference161=base))
        near=z['observed']&counteracts(z['real'],base)
    stable=np.ones(n,bool)
    for tag in ['shuffled','order_sorted']:
        with np.load(LATENT/f'{tag}.npz') as z:
            stable &= counteracts(z['p'][:n,1],base)
    retain=~(near|stable)
    eligible=common&retain
    wrong=y!=base
    assert (eligible.sum(),(eligible&wrong).sum(),(eligible&~wrong).sum())==(7055,855,6200)
    panels=counteracts(x,base)
    control,treatment=np.moveaxis(counteracts(v,base),2,0)
    introduced=treatment&~control
    withdrawn=control&~treatment
    new_in_archives=treatment.any(1)&~panels.any(1)
    all_opinions=np.column_stack([panels,control,treatment]).any(1)
    size=raw.groupby(g).PassengerId.transform('size').to_numpy()
    pay=raw[FEES].gt(0).any(axis=1).to_numpy()
    sleep=raw.CryoSleep.eq(True).to_numpy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    scopes=dict(retained=eligible,retained_new_orders=eligible&fresh,all_retained=known&retain,
                independent_new_orders=known&retain&fresh)
    masks=dict(whole_panel_reach=panels.any(1),any_control=control.any(1),any_treatment=treatment.any(1),
               any_opinion=all_opinions,no_opinion=~all_opinions,introduced= introduced.any(1),
               augmented_beyond_21=new_in_archives,added_beyond_21_and_all_controls=new_in_archives&~control.any(1),
               two_treatments=(treatment.sum(1)>=2),two_new_and_net_support=(introduced.sum(1)>=2)&(introduced.sum(1)>withdrawn.sum(1)),
               two_withdrawn_and_net_support=(withdrawn.sum(1)>=2)&(withdrawn.sum(1)>introduced.sum(1)))
    median=np.median(x,axis=1)>=.5
    exp={}
    for name,ordered in [('random',False),('contiguous',True)]:
        p=calibrate(x,v,median,median!=y,g,f,~fresh,ordered)
        exp[name]=p
        agree=(base==median)
        a,b=(p>=.5).T
        masks[name+'_original']=a&agree
        masks[name+'_augmented']=b&agree
        masks[name+'_extra']=b&~a&agree
    masks['both_protocols_extra']=masks['random_extra']&masks['contiguous_extra']
    tables={}
    for tag,scope in scopes.items():
        tables[tag]=dict(rows=int(scope.sum()),wrong=int((scope&wrong).sum()),correct=int((scope&~wrong).sum()),
                         rules={name:tally(a,wrong,scope,f) for name,a in masks.items()})
    individual={}
    for j,name in enumerate(names):
        individual[str(name)]={s:tally(panels[:,j],wrong,m,f) for s,m in list(scopes.items())[:2]}
    for j,name in enumerate(models):
        individual[name]={s:{key:tally(a,wrong,m,f) for key,a in
                            [('control',control[:,j]),('treatment',treatment[:,j]),('introduced',introduced[:,j]),
                             ('treatment_alone_vs_21',treatment[:,j]&~panels.any(1)),
                             ('unique_vs_other_treatments_and_21',new_in_archives&treatment[:,j]&(treatment.sum(1)==1))]}
                          for s,m in list(scopes.items())[:2]}
    mechanism_null={}
    for tag,scope in list(scopes.items())[:2]:
        gains=[]
        for orientation in product((False,True),repeat=len(models)):
            swap=np.array(orientation)
            added=np.where(swap,withdrawn,introduced).sum(1)
            removed=np.where(swap,introduced,withdrawn).sum(1)
            gains.append(tally((added>=2)&(added>removed),wrong,scope,f)['net'])
        mechanism_null[tag]=dict(actual=gains[0],median=float(np.median(gains)),lo=min(gains),hi=max(gains),
                                  orientations_at_least_actual=int(np.sum(np.asarray(gains)>=gains[0])),total=len(gains))
    inventory={}
    for state,flag in [('awake_paid',awake&pay),('sleep_unpaid',sleep&~pay),('other',~((awake&pay)|(sleep&~pay)))]:
        for direction in (False,True):
            a=eligible&flag&(base==direction)
            inventory[f'{state}_predicted_{direction}']=dict(rows=int(a.sum()),wrong=int((a&wrong).sum()),
                  solo_wrong=int((a&wrong&(size==1)).sum()), wrong_with_panel_dissent=int((a&wrong&panels.any(1)).sum()),
                  wrong_with_new_architecture_only=int((a&wrong&new_in_archives).sum()),
                  wrong_with_no_opinion=int((a&wrong&~all_opinions).sum()))
    assert sum(z['wrong'] for z in inventory.values())==855
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,reference161=base,eligible=eligible,
        known=known,fresh=fresh,names=np.array(models),pairs=v,original_panel=x,**exp)
    result=dict(methods=models,scope='frozen 11+161 agreement and no challenge by Nori adjacency or both difficulty splits together',
        protocol='Fixed matched-control crossings. Compare a non-selected panel summary and five unmodified controls to adding the five paired log-odds changes. Fit exclusively on historical bookings within each source-model outer holdout and exclude the calibration query order. The 855 retrospective misses never select a prediction.',
        tables=tables,individual=individual,orientation_reference=mechanism_null,inventory=inventory)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(tables['retained']['rules']['both_protocols_extra'],tables['retained_new_orders']['rules']['both_protocols_extra'])


if __name__=='__main__':
    run()
