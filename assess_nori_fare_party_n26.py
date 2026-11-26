"""Check nested fare features on frozen champion-plus-111 decisions, including every common mistake."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import log_loss

from arbitration_against_scored_nine_n26 import champions
from exhaustive_family_quorums_n26 import inputs
from held_expert_portfolios_n26 import DECK,SIX,panel_predictions,sources
from nori_fare_party_n26 import ART,FEES,REPORT,SHA,data,packet_dir
from rebase_fare_six_n26 import predict as fixed_recipe


def voting(anchor,old,own,peer,eligible,count):
    old_ok=(old>=.5)==anchor
    own_opposes=(own>=.5)!=anchor
    peer_opposes=(peer>=.5)!=anchor
    own_change=eligible&old_ok&own_opposes
    peer_change=(count>0)&old_ok&~own_opposes&peer_opposes
    return dict(own=own_change,independent_peer=peer_change,union=own_change|peer_change,
                both_models=eligible&old_ok&own_opposes&peer_opposes,
                peer_confirms_own=eligible&(count>0)&old_ok&own_opposes&peer_opposes)


def counts(act,y,anchor,mask,f):
    fixes=mask&act&(anchor!=y)
    harms=mask&act&(anchor==y)
    changes=fixes.astype(int)-harms
    return dict(rows=int(mask.sum()),errors=int((mask&(anchor!=y)).sum()),fixed=int(fixes.sum()),
                harmed=int(harms.sum()),net=int(changes.sum()),fold_net=np.bincount(f,weights=changes).astype(int).tolist())


def metrics(p,y,mask):
    z=p[mask].clip(1e-6,1-1e-6)
    return dict(rows=int(mask.sum()),accuracy=float(np.mean((z>=.5)==y[mask])),
                brier=float(np.mean((z-y[mask])**2)),logloss=float(log_loss(y[mask],z,labels=[0,1])))


def run():
    raw,y,g,f=data()
    _,cur,_=inputs()
    _,yy,gg,views=sources()
    for a,b in [(y,yy),(g,gg)]:np.testing.assert_array_equal(a,b)
    nine,known,_,hi=champions()
    bank=views['full']
    p=panel_predictions(bank,'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
    frozen=fixed_recipe(nine,p['neural_deck_raw'],p['neural_deck_fine'],
                        np.column_stack([p[key] for key in SIX]),bank['votes'][:,:18],eligible)['joint']>=.5
    truth=np.asarray(y,bool)
    assert known.sum()==8532
    check=counts(frozen!=nine,truth,nine,known,f)
    assert (check['fixed'],check['harmed'],check['net'])==(146,79,67)
    scores=np.full((len(y),3),np.nan)
    peer_count=np.zeros(len(y),int)
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        for j,name in enumerate(['raw','own','peer']):
            path=packet_dir(0)/f'fold{k}.npz' if j==0 else ART/f'{name}{k}.npz'
            with np.load(path) as z:
                np.testing.assert_array_equal(z['t'],t)
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['ids' if j==0 else 'passenger'],raw.PassengerId.iloc[q].to_numpy(str))
                np.testing.assert_array_equal(z['labels'],truth[t])
                assert str(z['checkpoint' if j==0 else 'sha'])==SHA
                scores[q,j]=z['p']
                if j:
                    np.testing.assert_array_equal(eligible[q],z['own_evidence'])
                    if j==1:peer_count[q]=z['peer_count']
                    else:np.testing.assert_array_equal(peer_count[q],z['peer_count'])
    old,own,peer=scores.T
    fees=raw[FEES]
    paid=fees.gt(0).any(axis=1).to_numpy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    masks=dict(all=np.ones(len(y),bool),awake_paid=awake&paid,no_positive=~paid,
               all_fees_recorded_zero=fees.eq(0).all(axis=1).to_numpy(),common=cur['common'],
               no_positive_with_paying_peer=(~paid)&(peer_count>0),new_orders=~np.isin(g,g[hi]))
    actions=voting(frozen,old,own,peer,eligible,peer_count)
    from full_roster_booking_validation_n23 import OUT as validation
    with np.load(validation/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        margin=(2*frozen.astype(float)-1)*np.abs(logit(z['p'].clip(.005,.995)))
    raw_log,own_log,peer_log=logit(scores.clip(.005,.995)).T
    own_delta=np.where(eligible,own_log-raw_log,0.)
    peer_delta=np.where(peer_count>0,peer_log-own_log,0.)
    actions['own_log_odds_shift']=((margin+own_delta>=0)!=frozen)&(own_delta!=0)
    actions['peer_log_odds_shift']=((margin+peer_delta>=0)!=frozen)&(peer_delta!=0)
    actions['combined_log_odds_shift']=((margin+own_delta+peer_delta>=0)!=frozen)&((own_delta+peer_delta)!=0)

    masks.update(paying_peer=peer_count>0,positive_bill_complete=eligible,
                 common_nonpayer_with_paying_peer=cur['common']&~paid&(peer_count>0))
    results={}
    for scope,mask in masks.items():
        available=mask&known
        results[scope]=dict(predictive_quality={n:metrics(s,truth,mask) for n,s in zip(('raw','own','peer'),scores.T)},
            versus_111={n:counts(a,truth,frozen,available,f) for n,a in actions.items()},
            raw_standalone={n:counts((s>=.5)!=frozen,truth,frozen,available,f) for n,s in zip(('raw','own','peer'),scores.T)},
            peer_versus_own=counts((peer>=.5)!=(own>=.5),truth,own>=.5,mask,f),
            changed_versus_raw={n:counts((s>=.5)!=(old>=.5),truth,old>=.5,mask,f) for n,s in [('own',own),('peer',peer)]})
    latest=cur['parent']
    results['common']['against_previous_eleven']={n:counts((s>=.5)!=latest,truth,latest,cur['common'],f)
                                                   for n,s in zip(('raw','own','peer'),scores.T)}
    np.savez_compressed(ART/'aligned.npz',passenger=raw.PassengerId.to_numpy(str),row=np.arange(len(y)),fold=f,y=truth,
                        raw=old,own=own,peer=peer,peer_count=peer_count,eligible=eligible,champion111=frozen,known=known)
    test_path=ART.parent.parent/'data/raw/test.csv'
    query=pd.read_csv(test_path,usecols=['PassengerId',*FEES])
    qfee=query[FEES]
    qelig=qfee.notna().all(axis=1)&qfee.gt(0).any(axis=1)
    qgroup=query.PassengerId.str.split('_').str[0]
    qpeer=(qelig.groupby(qgroup).transform('sum')-qelig.astype(int))>0
    report=dict(test_feature_support=dict(rows=len(query),own=int(qelig.sum()),peer=int(qpeer.sum()),
        nonpayer_with_paying_peer=int((qpeer&~qfee.gt(0).any(axis=1)).sum())),checkpoint=SHA,features=['eight inferred own-deck probabilities','inferred probability of registered deck',
        'number of other members with complete positive bills','mean eight-deck posterior among eligible other members'],
        split='Eight held booking folds; five inner booking folds for training-row fare densities. No Transported labels in posterior inference.',
        known_rows=int(known.sum()),baseline_accuracy=float(np.mean(frozen[known]==truth[known])),scopes=results)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(results['all']['predictive_quality'],results['all']['versus_111'],flush=True)


if __name__=='__main__':
    run()
