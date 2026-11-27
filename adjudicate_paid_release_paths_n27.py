"""Use independent archive mechanisms to review accepted and declined changes to the fare champion."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from family_identity_transfer_n26 import panels_and_reference
from held_expert_portfolios_n26 import SIX,DECK,panel_predictions,sources
from kaggle_workflow import sha256
from rebase_fare_six_n26 import NAME as BASE,predict
from repeated_archive_misses_n26 import ART,ROOT,cohort,count
from full_roster_booking_validation_n23 import data

OUT=ART/'nov27_adjudicate_paid_release_paths'
REPORT=ROOT/'reports/nov23_matched_learning_curve/adjudicate_paid_release_paths.json'
# Four distinct mechanisms and their matched controls, each a frozen model from the same archive.
PAIRS=[('companion_transition/real','companion_transition/swapped'),
       ('fixed_bill_margins/measured','fixed_bill_margins/same_fit'),
       ('class_neighborhood_mass/density','class_neighborhood_mass/shuffled'),
       ('permutation_selection/protected_select','permutation_selection/protected_random')]


def design(nine,raw,fine,six,fam,eligible,archive,names):
    paths=predict(nine,raw,fine,six,fam,eligible)
    base=paths['joint']>=.5
    accepted=base!=nine
    alternate=(fine>=.5)!=base
    sixth=(six>=.5)!=base[:,None]
    # Every accepted reversal is reviewable. An omitted proposal must have either fare or a six-expert majority.
    challenge=eligible&(accepted|alternate|(sixth.sum(1)>=4))
    geometry=np.column_stack([nine,base,paths['fare']>=.5,paths['six_alone']>=.5,raw,fine,fine-raw,
                              np.mean(six,1),sixth.sum(1)/6,np.mean(fam,1),np.std(six,1)])
    real,sham=[np.column_stack([archive[:,names.index(pair[j])] for pair in PAIRS]) for j in (0,1)]
    anchor_odds=np.where(base,1.,-1.)*abs(logit(fine.clip(.01,.99)))
    x={key:np.column_stack([geometry,extra]) for key,extra in [('control',sham),('real',real),
        ('contrast',np.column_stack([sham,real-sham]))]}
    x['paths']=geometry
    oppose=(real>=.5)!=base[:,None]
    stronger=(real-sham)*np.where(base,-1.,1.)[:,None]>0
    fixed={'all_four':challenge&oppose.all(1),'majority_four':challenge&(oppose.sum(1)>=3),
           'majority_stronger':challenge&((oppose&stronger).sum(1)>=3)}
    return dict(base=base,accepted=accepted,scope=challenge,x=x,prior=anchor_odds,fixed=fixed)


def learn(src,t,dst,q,key,y):
    a=src['x'][key][t]
    med=a.mean(0)
    sd=a.std(0).clip(.08)
    model=LogisticRegression(C=.12,max_iter=400).fit((a-med)/sd,y[t])
    return model.decision_function((dst['x'][key][q]-med)/sd)


def actions(d,scores):
    base=d['base']
    flags={k:d['scope']&((p>=0)!=base) for k,p in scores.items()}
    for key,ref in [('contrast','control'),('real','control'),('real','paths')]:
        flags[key+'_vs_'+ref]=d['scope']&((d['prior']+scores[key]-scores[ref]>=0)!=base)&flags[key]
    flags['confirmed']=flags['contrast_vs_control']&flags['real_vs_control']&flags['real_vs_paths']
    return {**d['fixed'],**flags}


def run():
    raw,y,g,f=data(); n=len(y)
    base161,blind,known,fresh=cohort(raw,y,f)
    old,identified,history,rows=champions(); np.testing.assert_array_equal(known,identified)
    yy,gg,ff,hr,names,p,h,*_=panels_and_reference()
    for a,b in [(yy,y),(ff,f),(hr,rows)]:np.testing.assert_array_equal(a,b)
    with np.load(DECK/'deck_estimates.npz') as z:eligible=z['eligible']
    _,_,_,views=sources()
    banks={}
    for tag,v in views.items():
        ix=v['ids']; panel=panel_predictions(v,tag)
        a=(old if tag=='full' else history)[ix]
        banks[tag]=design(a,panel['neural_deck_raw'],panel['neural_deck_fine'],np.column_stack([panel[x] for x in SIX]),
                          v['votes'][:,:18],eligible[ix],p if tag=='full' else h,names)
    test=pd.read_csv(ART/BASE/'submission.csv')
    frozen=ART/'proposal_n26_dispute_band/frozen_votes.npz'
    assert sha256(frozen)==json.loads((ROOT/'reports/proposal_n26_dispute_band/manifest.json').read_text())['model_sha256']
    with np.load(frozen) as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        np.testing.assert_array_equal(z['names'],names)
        pa=z['p']
    with np.load(ART/BASE/'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test.PassengerId)
        testbank=design(z['anchor'],z['raw'],z['fine'],z['challengers'],z['family_votes'],z['eligible'],pa,names)
    np.testing.assert_array_equal(testbank['base'],test.Transported)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:n],raw.PassengerId)
        np.testing.assert_array_equal(z['names'][-3:],['Mitra','TabICL fare','Nori fare'])
        med=np.median(z['scores'][:n,-3:],axis=1)
    first=banks['full']['base']
    risk=np.where(first,1-med,med)
    direction=np.zeros(n,bool)
    halves=partition(g,'directional_foundation',4)
    for k in np.unique(f):
        for j in range(4):
            t=np.flatnonzero(known&(f==k)&(halves!=j));q=np.flatnonzero((f==k)&(halves==j))
            cut=calibrate(risk,first,y,t,1.)
            direction[q]=risk[q]>=cut[first[q].astype(int)]
    registered=pd.read_csv(ART/'proposal_n27_directional_foundation/submission.csv')
    np.testing.assert_array_equal(registered.PassengerId,test.PassengerId)
    changed=registered.Transported.to_numpy()!=testbank['base']
    assert changed.sum()==16
    answer={}; test_acts={}; train_acts={}; history_apply={}
    for tag,d in banks.items():
        ix=views[tag]['ids']; fold=views[tag]['fold']; truth=y[ix]; groups=g[ix]
        valid=known[ix] if tag=='full' else np.ones(len(ix),bool)
        scopes=dict(all=valid,reverse=valid&d['accepted'],new=valid&~d['accepted'])
        if tag=='full':scopes.update(fresh=valid&fresh,blind=blind&(base161==d['base']))
        runs={}
        for split in ['random','ordered']:
            cells=partition(groups,'adjudicate_paths_n27',4)
            estimates={k:np.zeros(len(ix)) for k in d['x']}
            test_est={k:[] for k in d['x']}
            for k in np.unique(fold):
                if split=='ordered':
                    for j,orders in enumerate(np.array_split(np.unique(groups[fold==k]),4)):cells[(fold==k)&np.isin(groups,orders)]=j
                for j in [0,1,2,3,None]:
                    if j is None and split=='ordered':continue
                    t=np.flatnonzero((fold==k)&valid&d['scope']&((cells!=j) if j is not None else True))
                    q=np.flatnonzero((fold==k)&(cells==j)) if j is not None else np.arange(len(test))
                    if j is not None:assert not np.isin(groups[t],groups[q]).any()
                    for key in estimates:
                        s=learn(d,t,d if j is not None else testbank,q,key,truth)
                        if j is None:test_est[key].append(s)
                        else:estimates[key][q]=s
            flags=actions(d,estimates)
            runs[split]={}
            if split=='random':test_acts[tag]=actions(testbank,{k:np.mean(v,axis=0) for k,v in test_est.items()})
            train_acts[tag+'_'+split]=flags
            for key,flag in flags.items():
                delta=flag*np.where(d['base']!=truth,1,-1)
                runs[split][key]=dict(scopes={s:count(flag,m,d['base']!=truth) for s,m in scopes.items()},
                                      folds=[int(delta[valid&(fold==k)].sum()) for k in np.unique(fold)],
                                      test_changes=int(test_acts[tag][key].sum()))
        answer[tag]=runs
    old=banks['historical'];curr=banks['full']
    ht=y[rows]; oldfold=views['historical']['fold']
    # Historical calibration never sees the outcomes of entirely fresh bookings.
    for key in old['x']:
        p=[]
        for k in np.unique(oldfold):
            t=np.flatnonzero(old['scope']&(oldfold==k))
            p.append(learn(old,t,curr,np.arange(n),key,ht))
        history_apply[key]=np.mean(p,axis=0)
    filtered=actions(curr,history_apply)
    # For official row actions, require the same rule to generalize to both training-size protocols.
    frozen_actions={key:test_acts['historical'][key]&test_acts['full'][key] for key in test_acts['full']}
    answer['against_16']={}
    variants={}
    for key,test_act in frozen_actions.items():
        for domain in ['any','reverse','new']:
            test_region={'any':np.ones(len(test),bool),'reverse':testbank['accepted'],'new':~testbank['accepted']}[domain]
            local_region={'any':np.ones(n,bool),'reverse':curr['accepted'],'new':~curr['accepted']}[domain]
            pick=test_act&test_region
            extra=pick&~changed
            tag=key+'_'+domain
            variants[tag]=extra
            entry=dict(test_changes=int(extra.sum()),overlap_with_16=int((pick&changed).sum()))
            for split in ['random','ordered']:
                curact=train_acts['full_'+split][key]&local_region&~direction
                oldact=train_acts['historical_'+split][key]
                same=curact[rows]&oldact&(first[rows]==old['base'])
                repeat_ok=known[rows]
                unseenc=curact & filtered[key] & fresh & known
                entry[split]=dict(all=count(curact,known,first!=y),
                    historical_repeat=count(same,repeat_ok,first[rows]!=y[rows]),
                    fresh=count(curact,fresh&known,first!=y),fresh_history_transfer=count(unseenc,known,first!=y),
                    blind=count(curact,blind&(first==base161),first!=y))
            answer['against_16'][tag]=entry
    answer['consensus_test_edits']={k:int(v.sum()) for k,v in frozen_actions.items()}
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'test_actions.npz',passenger=test.PassengerId.to_numpy(str),base=testbank['base'],**frozen_actions)
    np.savez_compressed(OUT/'against16.npz',passenger=test.PassengerId.to_numpy(str),base=registered.Transported.to_numpy(),**variants)
    for tag,act in test_acts.items():
        np.savez_compressed(OUT/('test_'+tag+'.npz'),passenger=test.PassengerId.to_numpy(str),base=testbank['base'],**act)
    for tag,act in train_acts.items():np.savez_compressed(OUT/(tag+'.npz'),**act)
    REPORT.write_text(json.dumps(answer,indent=2)+'\n')
    for key in frozen_actions:
        print(key,[(tag,split,(r:=answer[tag][split][key]['scopes']['all'])['fixed']-r['harmed']) for tag in banks for split in ['random','ordered']],answer['consensus_test_edits'][key])


if __name__=='__main__':
    with threadpool_limits(1):run()
