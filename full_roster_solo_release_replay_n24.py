"""Replay two solo alternatives with release-compatible probabilities and fresh booking folds."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from export_budget_quarter_n23 import source
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_fixed_bill_margins_n24 import matrices
from full_roster_gate_probability_n23 import gate_probabilities
from kaggle_workflow import sha256
from full_roster_permutation_selection_n24 import cat,ensemble,feature_bank
from full_roster_selected_tree_replay_n24 import original_prediction,prepared_vote
from full_roster_solo_coverage_n24 import cached


def decide(base,solo,challenger,p):
    return np.where(solo & ((p>=.5)==base),.75*p+.25*challenger>=.5,base)


def effects(pred,baseline,y,mask):
    return [int(((pred==y)&(baseline!=y)&mask).sum()),int(((pred!=y)&(baseline==y)&mask).sum())]


def run():
    raw,y,g,fold = data()
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_ = aligned()
    names,h,full = banks(raw,y,fold)
    ref = latest_reference(y,g,fold,names,h,full)
    gate,_ = gate_probabilities(names,h,full,y,g,fold)
    np.testing.assert_array_equal(gate>=.5,ref)
    ix = np.arange(len(y))
    art = ROOT/'artifacts'
    mitra = full[:,names.index('mitra96')]
    solo = raw.groupby(g).PassengerId.transform('size').to_numpy()==1
    solo_old = solo[rows]
    unseen = ~np.isin(g,og)
    opinions = full[:,[names.index(s) for s in (*FAMILIES,'mitra96',*EVIDENCE)]]
    nine = prepared_vote(original_prediction(ix,fold,g,y,'full'),ref,opinions)
    measured = matrices(raw,g,raw[['Spa','FoodCourt','VRDeck','RoomService','ShoppingMall']].fillna(0).to_numpy())[2]
    wide,fields,_ = feature_bank(raw)
    cols = np.random.default_rng(1216).permutation(len(fields))[:52]
    x = wide[:,cols]
    with np.load(art/'nov24_remaining_cases/archive_scores.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('row',rows),('old_fold',of),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        j = z['names'].tolist().index('permutation_selection/random')
        old_random,full_random = z['historical'][:,j],z['full'][:,j]
    a = np.column_stack([cached('fixed_bill_margins','historical',rows,of,g,y)[:,2],old_random])
    b = np.column_stack([cached('fixed_bill_margins','full',ix,fold,g,y)[:,2],full_random])
    # Reproduce both original fold-0 predictions before training on new assignments.
    t,q = ix[fold!=0],ix[fold==0]
    np.testing.assert_array_equal(cat(497).fit(measured[t],y[t]).predict_proba(measured[q],thread_count=3)[:,1],b[q,0])
    np.testing.assert_array_equal(ensemble(x,y,t,q,497),b[q,1])
    out = art/'nov24_solo_release_replay'
    out.mkdir(exist_ok=True)
    models = ['measured','random_fields']
    report = {'historical':{mode:effects(decide(champ,solo_old,a[:,j],old[:,0]),champ,oy,solo_old)
                            for j,mode in enumerate(models)}}
    for seed in (0,1021,1022):
        refit_nine = nine
        if seed:
            v = np.full_like(b,np.nan)
            tree_scores = np.full(len(y),np.nan)
            for f,(t,q) in enumerate(GroupKFold(5,shuffle=True,random_state=seed).split(x,groups=g)):
                assert not np.isin(g[t],g[q]).any()
                with np.load(art/'nov24_selected_tree_replay'/f'{seed}_{f}.npz') as z:
                    for k,val in [('t',t),('q',q),('truth',y[t])]:
                        np.testing.assert_array_equal(z[k],val)
                    tree_scores[q] = z['p']
                path = out/f'{seed}_{f}.npz'
                if not path.exists():
                    prob = np.column_stack([cat(497).fit(measured[t],y[t]).predict_proba(measured[q],thread_count=3)[:,1],
                                            ensemble(x,y,t,q,497)])
                    with path.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream,p=prob,t=t,q=q,y=y[t])
                    path.with_suffix('.tmp').replace(path)
                with np.load(path) as z:
                    for k,val in [('t',t),('q',q),('y',y[t])]:
                        np.testing.assert_array_equal(z[k],val)
                    v[q] = z['p']
            assert np.isfinite(v).all() and np.isfinite(tree_scores).all()
            refit_nine = prepared_vote(tree_scores,ref,opinions)
        else:
            v = b
        for j,mode in enumerate(models):
            staging = decide(ref,solo,v[:,j],gate)
            release = decide(ref,solo,v[:,j],mitra)
            together = np.where(nine!=ref,nine,release)
            report[f'{seed}_{mode}'] = dict(calibrated_anchor=effects(staging,ref,y,solo),
                        release_anchor=effects(release,ref,y,solo),unseen=effects(release,ref,y,unseen),
                        joint=effects(together,ref,y,True),
                        both_refit=effects(np.where(refit_nine!=ref,refit_nine,release),ref,y,True),
                        nine_only=effects(refit_nine,ref,y,True),overlap=int(((nine!=ref)&(release!=ref)).sum()))
            print(seed,mode,report[f'{seed}_{mode}'],flush=True)
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda col:col!='Name')
    tg = test.PassengerId.str.split('_').str[0].to_numpy()
    st = test.groupby(tg).PassengerId.transform('size').to_numpy()==1
    for col in ('HomePlanet','Destination','VIP'):
        assert set(raw[col].dropna())==set(test[col].dropna())
    for j in (0,2):
        assert set(raw.Cabin.dropna().str.split('/').str[j])==set(test.Cabin.dropna().str.split('/').str[j])
    query = matrices(test,tg,test[['Spa','FoodCourt','VRDeck','RoomService','ShoppingMall']].fillna(0).to_numpy())[2]
    ptest = cat(497).fit(measured,y).predict_proba(query,thread_count=3)[:,1]
    anchor,_,_ = source('proposal_online_first_counts15',test)
    _,inc,_ = source('proposal_v131_party_consensus',test)
    _,random_decision,_ = source('proposal_n24_random_subspace',test)
    queue_dir = art/'proposal_n24_selected_dissent'
    manifest = json.loads((ROOT/'reports'/queue_dir.name/'manifest.json').read_text())
    assert sha256(queue_dir/'submission.csv')==manifest['sha256']
    queue_table = pd.read_csv(queue_dir/'submission.csv')
    np.testing.assert_array_equal(queue_table.PassengerId,test.PassengerId)
    queued = queue_table.Transported.to_numpy(bool)
    for mode,pred in [('measured',decide(inc,st,ptest,anchor)),('random_fields',np.where(st,random_decision,inc))]:
        changed = pred!=inc
        report['test_'+mode] = dict(changes=int(changed.sum()),to_true=int((pred&changed).sum()),
                 to_false=int((~pred&changed).sum()),overlap=int((changed&(queued!=inc)).sum()))
    path = ROOT/'reports/nov23_matched_learning_curve/solo_release_replay.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(report['test_measured'],report['test_random_fields'])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
