"""Select historic disagreements on one set of bookings; independently learn a veto for collateral errors."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from dissent_likelihood_transfer_n27 import splits

PARENTS={'direction':'proposal_n27_directional_foundation','three':'proposal_n27_three_payment_states',
         'union':'proposal_n27_receipt_mitra_party_union','average':'proposal_n27_complete_receipt_average'}
OUT=ART/'nov28_archive_source_harm_veto'
REPORT=ROOT/'reports/nov23_matched_learning_curve/archive_source_harm_veto.json'


def inputs():
    raw,y,g,f=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    frame=pd.concat([raw,test],ignore_index=True);ids=frame.PassengerId.to_numpy(str)
    def get(path,*fields):
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z['ids' if 'ids' in z else 'passenger'],ids)
            if 'y' in z:np.testing.assert_array_equal(z['y'],y)
            return [z[k] for k in fields]
    first,mask=get('nov27_dissent_likelihood_transfer/aligned.npz','first','random_median')
    anchor=first^mask
    mitra,=get('nov27_mitra_fare/aligned.npz','111_specific_with_support')
    _,_,_,_,names,p,*_=panels_and_reference()
    with np.load(ART/'proposal_n26_dispute_band/frozen_votes.npz') as z:
        np.testing.assert_array_equal(z['names'],names);np.testing.assert_array_equal(z['passenger'],ids[n:])
        p=np.vstack([p,z['p']])
    # Distinct, named historical treatments are the candidates; null controls are held aside.
    avoid=('shuffl','placebo','swapped','matched_random','reversed','in_fit_offset','protected_random')
    keep=[i for i,name in enumerate(names) if not any(word in name.lower() for word in avoid)]
    names=np.asarray(names)[keep];p=p[:,keep]
    wide,=get('nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz','opinions')
    official={}
    for key,name in PARENTS.items():
        csv=pd.read_csv(ART/name/'submission.csv');np.testing.assert_array_equal(csv.PassengerId,ids[n:])
        records=[json.loads(path.read_text()) for path in (ROOT/'reports'/name/'online_attempts').glob('*.json')]
        assert any(str(z['evaluation']['Public Score']) in ('0.82370','0.8237') and z['evaluation']['Status']=='COMPLETE' for z in records)
        official[key]=csv.Transported.to_numpy()
    versions={}
    for mode,em in [('random','random'),('ordered','contiguous')]:
        route,=get('nov27_payment_state_routes/aligned.npz',mode)
        a,b=get(f'nov27_complete_receipt_counterfactual/aligned_{mode}_2.npz','average_random_calibrated','average_ordered_calibrated')
        party,=get('nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz',mode)
        versions[mode]=np.column_stack([anchor,anchor^route.any(1),anchor^((mitra&(first==anchor))|(a&b)|party),anchor^(a if mode=='random' else b)])
        for j,k in enumerate(PARENTS):np.testing.assert_array_equal(versions[mode][n:,j],official[k])
    return frame,y,g,f,ids,names,p,wide,versions


def select_sources(bank,risk,signed,family,states):
    """Choose one candidate per family by its excess correct decisions net of a sample-size penalty."""
    selected={}
    for c in range(4):
        t=bank[states[bank]==c]
        proposals=signed[t]>0
        cases=proposals.sum(0)
        delta=proposals.T.astype(float)@np.where(risk[t],1.,-1.)
        # Shrunken precision resists selecting a source that proposed a single lucky change.
        reward=(delta-2.)/(cases+10.)*np.sqrt(np.minimum(cases,40)/40)
        picks=[]
        for value in np.unique(family):
            idx=np.flatnonzero(family==value)
            j=idx[np.argmax(reward[idx])]
            if cases[j]>=3:picks.append(j)
        selected[c]=np.array(sorted(picks,key=lambda j:-reward[j])[:4],int)
    return selected


def candidate(rows,selected,signed,wide,states,aux):
    features=np.zeros((len(rows),9))
    for c,ix in selected.items():
        m=states[rows]==c
        if not len(ix) or not m.any():continue
        z=np.sort(signed[rows[m]][:,ix],axis=1)[:,::-1]
        z=np.pad(z,((0,0),(0,4-z.shape[1])),constant_values=-.5)
        features[m]=np.column_stack([(z>0).mean(1),z[:,0],z[:,1],z[:,2],wide[rows[m]].max(1),
                                    wide[rows[m]].mean(1),aux[rows[m]],np.full(m.sum(),c//2),np.full(m.sum(),c%2)])
    return features


def veto(train,targets,query):
    active=train[:,0]>0
    if active.sum()<12 or len(np.unique(targets[active]))<2:return np.zeros(len(query))
    mean=train[active].mean(0);scale=train[active].std(0).clip(.10)
    model=LogisticRegression(C=.22,max_iter=400).fit((train[active]-mean)/scale,targets[active])
    return np.where(query[:,0]>0,model.predict_proba((query-mean)/scale)[:,1],0.)


def run():
    frame,y,g,f,ids,names,archive,seven,versions=inputs();n=len(y)
    fixed=versions['random'][:,0]
    _,blind,known,fresh=cohort(frame.iloc[:n],y,f)
    signed=(archive-.5)*np.where(fixed,-1.,1.)[:,None]
    ordinary=(seven-.5)*np.where(fixed,-1.,1.)[:,None]
    family=np.array([k.split('/')[0] for k in names]);same=((seven>=.5)==fixed[:,None]).all(1)
    assert (known&same[:n]&(fixed[:n]!=y)).sum()==862
    state=2*same.astype(int)+fixed.astype(int)
    spend=frame[['VRDeck','Spa','ShoppingMall','FoodCourt','RoomService']].gt(0).any(axis=1).to_numpy()
    wrong=fixed[:n]!=y
    test_groups=frame.PassengerId.iloc[n:].str[:4].to_numpy()
    tg=np.array([int.from_bytes(hashlib.sha256(('source-evaluation-'+v).encode()).digest()[:4],'big') for v in test_groups],dtype=np.int64)
    sorted_test=np.unique(test_groups)
    travel=np.searchsorted(sorted_test,test_groups)*4//len(sorted_test)
    allq=np.arange(n,len(ids));options=['any_selected','two_sources','veto','two_veto']
    result={}; output={};OUT.mkdir(exist_ok=True)
    for mode,em in [('random','random'),('ordered','contiguous')]:
        fold=splits(g,f,em)
        equal=(versions[mode]==fixed[:,None]).all(1)
        valid=known&equal[:n]
        eval_masks={'all':known,'historically_new':known&fresh,'seven_and_four_unchanged':valid&same[:n],'named_hard':blind}
        calls={k:np.zeros(len(ids),bool) for k in options}
        test_votes={k:[] for k in options};uses=np.zeros((4,len(names)),int)
        for k in range(8):
            for j in range(4):
                q=np.flatnonzero((f==k)&(fold==j))
                assigned=(tg%8==k)&(((tg//8)%4 if em=='random' else travel)==j)
                test_rows=allq[assigned]
                query=np.r_[q,test_rows]
                ballots={label:[] for label in options}
                for verifier in (1,2,3):
                    held=(j+verifier)%4
                    discover=np.flatnonzero((f==k)&valid&(fold!=j)&(fold!=held))
                    calibrate=np.flatnonzero((f==k)&valid&(fold==held))
                    assert not np.isin(g[discover],np.r_[g[q],g[calibrate]]).any() and not np.isin(g[calibrate],g[q]).any()
                    selected=select_sources(discover,wrong,signed,family,state)
                    for c,ix in selected.items():uses[c,ix]+=1
                    xt=candidate(calibrate,selected,signed,ordinary,state,spend)
                    xq=candidate(query,selected,signed,ordinary,state,spend)
                    risk=veto(xt,wrong[calibrate],xq)
                    sets={'any_selected':xq[:,0]>0,'two_sources':xq[:,0]>=.5,'veto':risk>.5,'two_veto':(xq[:,0]>=.5)&(risk>.5)}
                    for label,proposal in sets.items():ballots[label].append(proposal)
                for label,proposals in ballots.items():
                    support=np.mean(proposals,axis=0)
                    calls[label][query]=(support>.5)&equal[query]
                    test_votes[label].append((test_rows-n,support[len(q):]))
        for key in list(calls):
            test_support=np.full(len(ids)-n,np.nan)
            for row,values in test_votes[key]:test_support[row]=values
            assert np.isfinite(test_support).all()
            np.testing.assert_array_equal(calls[key][n:],(test_support>.5)&equal[n:])
            output[mode+'_'+key+'_test_support']=test_support
            calls[key+'_seven']=calls[key]&same
        table={}
        for label,proposal in calls.items():
            output[mode+'_'+label]=proposal
            s={key:count(proposal[:n],m,wrong) for key,m in eval_masks.items()}
            d=proposal[:n]*np.where(wrong,1,-1)
            table[label]=dict(scopes=s,accuracy=float(np.mean((fixed[:n]^proposal[:n])[known]==y[known])),
                test=int(proposal[n:].sum()),seven_test=int((proposal[n:]&same[n:]).sum()),random_outer_fold_net=[int(d[(f==i)&known].sum()) for i in range(8)])
        result[mode]={'sources_discovered':{str(i):{str(names[j]):int(v[j]) for j in np.argsort(v)[-8:][::-1] if v[j]>0} for i,v in enumerate(uses)},
                      'rules':table}
    result['coverage']={}
    for k,m in [('all',known),('seven',known&same[:n]),('seven_four',known&same[:n]&(versions['random'][:n]==fixed[:n,None]).all(1))]:
        opp=(signed[:n]>0).any(1)&m
        result['coverage'][k]=dict(people=int(m.sum()),errors=int((m&wrong).sum()),reachable_correct=int((opp&wrong).sum()),possible_harm=int((opp&~wrong).sum()))
    same4=(versions['random'][:n]==fixed[:n,None]).all(1)
    result['individual_source_coverage']={}
    for k,name in enumerate(names):
        m=known&same[:n]&same4
        result['individual_source_coverage'][str(name)]=dict(**count(signed[:n,k]>0,m,wrong),test_seven=int(((signed[n:,k]>0)&same[n:]).sum()))
    report=dict(matched_application='Every held booking and assigned test booking averages three alternate discovery/veto splits. Never use abstaining consensus of 32 policies for test only.',parent=PARENTS,sources=len(names),families=len(np.unique(family)),known=int(known.sum()),parent_accuracy=float((~wrong[known]).mean()),
        limitation='Only source selection and the falsification veto are re-fit across the two within-fold booking partitions. Historic base outputs stay on their original, common booking folds. Four scored releases must agree. Labels outside each frozen outer fold are never used by the selector.',
        results=result)
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,y=y,fold=f,parent=fixed,seven=same,**output)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for key in options:
        for rule in [key,key+'_seven']:
            print(rule,[(mode,result[mode]['rules'][rule]['scopes']['all'],result[mode]['rules'][rule]['scopes']['seven_and_four_unchanged'],result[mode]['rules'][rule]['test']) for mode in versions])
    print(result['coverage'])

if __name__=='__main__':
    with threadpool_limits(1):run()
