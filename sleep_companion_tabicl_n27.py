"""Compare actual and displaced passenger companions in booking-held TabICL predictions."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from exaone_validation_data import compact_features
from matched_state_context_n26 import inputs, FEES, ROOT
from repeated_archive_misses_n26 import ART, cohort, count

OUT=ART/'nov27_sleeper_companion_tabicl'
REPORT=ROOT/'reports/nov23_matched_learning_curve/sleep_companion_tabicl_n27.json'


def views(frame, groups):
    x=compact_features(frame)
    slot=frame.PassengerId.str.split('_').str[-1].astype(int).to_numpy()
    cabins=frame.Cabin.fillna('').to_numpy(str)
    origins=x.HomePlanet.to_numpy(str)
    dest=x.Destination.to_numpy(str)
    deck=x.Deck.to_numpy(str)
    side=x.Side.to_numpy(str)
    age=x.Age.to_numpy(float)
    sleep=frame.CryoSleep.map({True:1.,False:0.}).to_numpy(float)
    bills=frame[FEES].to_numpy(float)
    luxury=np.log1p(np.nansum(frame[['VRDeck','Spa','RoomService']],axis=1))
    meals=np.log1p(np.nansum(frame[['FoodCourt','ShoppingMall']],axis=1))
    pay=np.nan_to_num(bills).sum(1)>0
    booking={group:np.asarray(idx) for group,idx in pd.Series(groups).groupby(groups).indices.items()}
    sizes=np.array([len(booking[g]) for g in groups])
    focus=(sleep==1)|~pay
    basic=x.copy()
    basic['booking_size']=sizes
    basic['booking_slot']=slot
    names=['same_cabin','same_origin','same_dest','same_deckside','awake_same_cabin','payer_same_cabin','payer_different_cabin',
           'matched_sleep','matched_luxury','matched_meals','matched_age','awake_missing_receipt']
    rng=np.random.default_rng(81102)
    unrelated={}
    for size in np.unique(sizes[sizes>1]):
        orders=[g for g,idx in booking.items() if len(idx)==size]
        # Match placebo parties on first registered origin and deck to avoid unrealistically random outsiders.
        rng.shuffle(orders)
        orders.sort(key=lambda g:(origins[booking[g][0]],deck[booking[g][0]]))
        assert len(orders)>1
        for g,h in zip(orders,orders[1:]+orders[:1]):
            unrelated[g]=h
    result={'plain':basic}
    for tag in ['real','placebo']:
        val=np.full((len(frame),len(names)),np.nan)
        for i in np.flatnonzero(sizes>1):
            other=booking[groups[i] if tag=='real' else unrelated[groups[i]]]
            other=other[slot[other]!=slot[i]]
            assert len(other)==sizes[i]-1 and i not in other
            same_cabin=(cabins[i]!='')&(cabins[other]==cabins[i])
            same_origin=(origins[i]!='__MISSING__')&(origins[other]==origins[i])
            same_dest=(dest[i]!='__MISSING__')&(dest[other]==dest[i])
            same_deckside=(deck[i]!='__MISSING__')&(deck[other]==deck[i])&(side[other]==side[i])
            # Weight the registered attributes of the companion with the closest match in cabin and route.
            affinity=4*same_cabin+2*same_deckside+same_origin.astype(int)+same_dest.astype(int)
            sel=other[affinity==affinity.max()]
            def mean_obs(a):
                seen=a[np.isfinite(a)]
                return seen.mean() if len(seen) else np.nan
            val[i]=[same_cabin.mean(),same_origin.mean(),same_dest.mean(),same_deckside.mean(),((sleep[other]==0)&same_cabin).mean(),
                    (pay[other]&same_cabin).mean(),(pay[other]&~same_cabin).mean(),mean_obs(sleep[sel]),luxury[sel].mean(),meals[sel].mean(),
                    mean_obs(age[sel]),((sleep[other]==0)&np.isnan(bills[other]).any(1)).mean()]
        result[tag]=pd.concat([basic,pd.DataFrame(val,columns=names)],axis=1)
    return result,focus,sizes


def train():
    import torch
    from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    frame,y,g,f,_=inputs()
    designs,focus,sizes=views(frame,g)
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(focus[:len(y)]&(f!=k))
        q=np.flatnonzero(focus[:len(y)]&(f==k)) if k<8 else np.flatnonzero(focus[len(y):])+len(y)
        assert not np.isin(g[t],g[q]).any()
        path=OUT/f'fold{k}.npz'
        if not path.exists():
            scores={key:infer(x,y,t,q,5805+k) for key,x in designs.items()}
            np.savez_compressed(path,t=t,q=q,y=y[t],ids=frame.PassengerId.iloc[q].to_numpy(str),checkpoint=DIGEST,**scores)
        with np.load(path) as z:
            for field,val in [('t',t),('q',q),('ids',frame.PassengerId.iloc[q]),('y',y[t]),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[field],val)
        print('finished',k,len(t),len(q),flush=True)



def score():
    from dissent_likelihood_transfer_n27 import splits
    from sklearn.linear_model import LogisticRegression
    from scipy.special import logit
    frame,y,g,f,scope=inputs()
    n=len(y); ids=frame.PassengerId.to_numpy(str)
    sleep=scope['sleep_unpaid']; focused=sleep|~frame[FEES].gt(0).any(axis=1).to_numpy()
    size=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    party=size>1
    _,blind,known,fresh=cohort(frame.iloc[:n],y,f)
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        anchor=z['parent']
        out=[np.median(z[k],axis=1) if z[k].ndim==2 else z[k] for k in ('dpt','tabnet')]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        historical=z['scores']
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        first=z['first']
        ref=z['first']^z['random_median']
    opinions=np.column_stack([np.median(historical[:,:18],axis=1), historical[:,-3:],
                              np.median(historical[:,-3:],axis=1),*out])
    shared=known&sleep[:n]&((opinions[:n]>=.5)==ref[:n,None]).all(1)
    assert (shared&(ref[:n]!=y)).sum()==300
    p={key:np.full(len(ids),np.nan) for key in ['plain','placebo','real']}
    for k in range(9):
        with np.load(OUT/f'fold{k}.npz') as z:
            q=z['q']
            np.testing.assert_array_equal(z['ids'],ids[q])
            assert (~np.isin(g[q],g[z['t']])).all()
            for key in p:
                p[key][q]=z[key]
    np.testing.assert_array_equal(np.isfinite(p['real']),focused)
    agreement=((opinions>=.5)==anchor[:,None]).all(axis=1)
    error=anchor[:n]!=y
    dep={'all':known,'focus':known&focused[:n],'fresh':known&fresh,'sleep':known&sleep[:n],
         'party':known&party[:n],'seven_sleep':shared,'seven_sleep_party':shared&party[:n]}
    inc=np.where(anchor,-1.,1.)
    opposing={key:focused&(((v>=.5)!=anchor)) for key,v in p.items()}
    rules={key:val&party for key,val in opposing.items()}
    rules['induced']=rules['real']&~rules['plain']
    rules['real_only']=rules['induced']&~rules['placebo']
    rules['confirmed']=rules['real']&rules['plain']
    rules['unopposed_control']=rules['real']&~rules['placebo']
    odds=lambda v:logit(np.clip(v,.01,.99))
    history=odds(np.median(opinions,axis=1))*inc
    raw=odds(p['plain'])*inc
    active=focused&party
    for kind in ('plain','placebo','real'):
        value=odds(p[kind])*inc
        x=np.column_stack([anchor,history,raw,value-raw,sleep,size>2])
        for protocol in ('random','contiguous'):
            partition=splits(g[:n],f,protocol)
            cross=np.full(len(ids),np.nan)
            tests=[]
            for k in range(8):
                for j in [0,1,2,3,None]:
                    t=np.flatnonzero(known&active[:n]&(f==k)&((partition!=j) if j is not None else True))
                    q=np.flatnonzero(active[:n]&(f==k)&(partition==j)) if j is not None else np.flatnonzero(active[n:])+n
                    assert not np.isin(g[t],g[q]).any()
                    center=x[t].mean(0);scale=x[t].std(0).clip(.20)
                    m=LogisticRegression(C=.12,max_iter=500).fit((x[t]-center)/scale,error[t])
                    risk=m.predict_proba((x[q]-center)/scale)[:,1]
                    if j is None:tests.append(risk)
                    else:cross[q]=risk
            cross[q]=np.mean(tests,axis=0)
            rules[kind+'_risk_'+protocol]=cross>.5
            rules[kind+'_change_'+protocol]=(cross>.5)&rules[kind]
    for key in ['risk','change']:
        for method in ('real','plain','placebo'):
            rules[method+'_'+key+'_both']=rules[method+'_'+key+'_random']&rules[method+'_'+key+'_contiguous']
        rules['novel_'+key]=rules['real_'+key+'_both']&~rules['plain_'+key+'_both']&~rules['placebo_'+key+'_both']
    for name in ('real','plain','placebo','induced','real_only','confirmed','real_risk_both','real_change_both'):
        rules[name+'_seven_sleep']=rules[name]&agreement&sleep
    nought=~frame[FEES].gt(0).any(axis=1).to_numpy()
    payer=pd.Series(~nought).groupby(g).transform('sum').to_numpy()-(~nought).astype(int)
    details={}
    for label,sub in [('solo',~party[:n]),('has_peer',party[:n]),('any_peer_paid',payer[:n]>0),
                      ('all_peers_unpaid',party[:n]&(payer[:n]==0)),('earth',frame.HomePlanet.eq('Earth').to_numpy()[:n])]:
        valid=shared&sub
        details[label]=dict(agree=int(valid.sum()),errors=int((valid&error).sum()),
            actually_transported=int((valid&error&y).sum()))
    duplicate_cabin=(pd.Series(g).groupby(frame.Cabin).transform('nunique')>1).to_numpy()
    result={'cross_booking_shared_cabin_rows':int(duplicate_cabin.sum()),'prior_controls':['joint_party_tabicl_n26.py','manifest_sleep_contrast_n26.py','peer_attribute_surprise_n23.py'],
            'support':'all target-scope passengers from training bookings; full 8-fold and independent test refit',
            'placebo':'rotate same-size bookings ordered after random tie breaking on first passenger origin and deck, keep within-party registration slot',
            'common_error_description':details,'rows':{k:int(m.sum()) for k,m in dep.items()},'errors':{k:int((m&error).sum()) for k,m in dep.items()},
            'reference_seven_errors':int((shared&(ref[:n]!=y)).sum()),
            'anchor_differences_seven':int((shared&(ref[:n]!=anchor[:n])).sum()),
            'party_seven_reference_errors':int((shared&(ref[:n]!=y)&party[:n]).sum()),'models':{},'rules':{}}
    for key,v in p.items():
        result['models'][key]={name:float(((v[:n][m]>=.5)==y[m]).mean()) for name,m in
            [('target',focused[:n]),('sleepers',sleep[:n]),('party_sleepers',party[:n]&sleep[:n]),('party',party[:n]&focused[:n])]}
    for key,a in rules.items():
        delta=np.where(error,1,-1)*a[:n]
        result['rules'][key]={'test':int(a[n:].sum()),'orders':int(len(np.unique(g[n:][a[n:]]))),
                              'scopes':{name:count(a[:n],m,error) for name,m in dep.items()},
                              'folds':[int(delta[known&(f==k)].sum()) for k in range(8)]}
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,fold=f,anchor=anchor,y=y,shared=shared,party=party,focused=focused,**p,
                        **{'rule_'+k:v for k,v in rules.items()})
    parent=pd.read_csv(ART/'proposal_n27_directional_foundation/test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(parent.PassengerId,ids[n:])
    np.testing.assert_array_equal(parent.probability>=.5,anchor[n:])
    complete=pd.DataFrame({'PassengerId':ids[n:],'target_scope':focused[n:],'has_peer':party[n:],'scored_parent':parent.probability})
    for key,values in p.items():
        complete[key]=np.where(party[n:]&focused[n:],values[n:],parent.probability)
    complete.to_csv(OUT/'test_probabilities.csv',index=False)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(result['models'],result['party_seven_reference_errors'],flush=True)
    for key,r in result['rules'].items():print(key,r['scopes']['all'],r['scopes']['fresh'],r['scopes']['seven_sleep'],r['test'])

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','score'),default='train',nargs='?')
    {'score':score,'train':train}[parser.parse_args().mode]()
