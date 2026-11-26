"""Bound both clean-fare releases on every passenger and check transfer across booking sizes."""
import hashlib
import json
import numpy as np
import pandas as pd

from export_consensus_review_n26 import ART, dig
from export_joint_fare_foundations_n26 import combine
from full_roster_booking_validation_n23 import ROOT, data
from grouped_peer_teacher_n25 import OUT as PEER
from held_expert_portfolios_n26 import DECK, SIX, panel_predictions, sources
from nori_fare_party_n26 import ART as NORI
from nori_booking_regression_n26 import packet_dir
from rebase_fare_six_n26 import predict


def bounds(delta, mask):
    d=delta[:,mask]
    lo,hi=d.min(0),d.max(0)
    return dict(rows=int(mask.sum()),net_bounds=[int(lo.sum()),int(hi.sum())],
                identified=int((lo==hi).sum()),possible_net_counts={
                    f'{a}..{b}':int(((lo==a)&(hi==b)).sum()) for a,b in [(0,0),(-1,-1),(1,1),(-1,0),(0,1),(-1,1)]})


def read_sources(people,y,g,fold):
    anchor=np.empty((2,len(y)),bool)
    old_prior=np.empty(len(y),int)
    nori=np.empty((len(y),2))
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        with np.load(PEER/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['y'],y[q])
            anchor[:,q]=[z['nine_on_low_branch'],z['nine_on_high_branch']]
            old_prior[q]=z['v131']
        for j,path in enumerate([packet_dir(0)/f'fold{k}.npz',NORI/f'own{k}.npz']):
            with np.load(path) as z:
                for n,value in [('t',t),('q',q),('labels',y[t]),(('ids','passenger')[j],people.PassengerId.iloc[q])]:
                    np.testing.assert_array_equal(z[n],value)
                nori[q,j]=z['p']
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['fold'],fold)
        np.testing.assert_array_equal(z['original_prior'],old_prior)
        prior=z['prior']
    known=prior>=0
    anchor[:,known]=anchor[prior[known],np.flatnonzero(known)]
    v=sources()[-1]['full']
    np.testing.assert_array_equal(v['ids'],np.arange(len(y)))
    p=panel_predictions(v,'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'][:len(y)],people.PassengerId)
        eligible=z['eligible'][:len(y)]
    scores=np.column_stack([p[n] for n in SIX])
    raw=np.column_stack([p['neural_deck_raw'],nori[:,0]])
    fine=np.column_stack([p['neural_deck_fine'],nori[:,1]])
    versions={'nine':anchor,'111':np.empty_like(anchor),'161':np.empty_like(anchor)}
    for idx,nine in enumerate(anchor):
        base=predict(nine,raw[:,0],fine[:,0],scores,v['votes'][:,:18],eligible)['joint']
        second,edits=combine(base,nine,eligible,raw,fine)
        versions['111'][idx]=base>=.5
        versions['161'][idx]=second>=.5
        assert not (edits&(versions['111'][idx]!=nine)).any()
    return known,versions


def sizes(groups):
    return pd.Series(groups).groupby(groups).transform('size').to_numpy()


def size_audit(delta,g,size,fresh,known,test_size):
    rows={}
    for name,train_mask,test_mask in [('1',size==1,test_size==1),('2',size==2,test_size==2),
                                      ('3',size==3,test_size==3),('4+',size>=4,test_size>=4)]:
        m=fresh&known&train_mask
        n=int(m.sum())
        rows[name]=dict(fresh_rows=n,fresh_bookings=len(set(g[m])),all_known=int((known&train_mask).sum()),
                        test_rows=int(test_mask.sum()),**bounds(delta,m))
    def standardized(train_bands,test_bands):
        result=0.
        available_weight=0.
        for s,t in zip(train_bands,test_bands,strict=True):
            mask=fresh&known&s
            if not mask.any(): continue
            weight=t.mean();available_weight+=weight
            result+=weight*delta[0,mask].mean()
        return dict(supported_target_share=float(available_weight),delta_pp_on_supported_target=100*result/available_weight)
    return dict(size_cells=rows,solo_vs_party_standardized=standardized([size==1,size>1],[test_size==1,test_size>1]),
                four_size_standardized=standardized([size==1,size==2,size==3,size>=4],[test_size==1,test_size==2,test_size==3,test_size>=4]))


def main():
    people,y,g,fold=data()
    known,versions=read_sources(people,y,g,fold)
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        fresh=~np.isin(g,g[z['rows']])
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    size,ts=sizes(g),sizes(tg)
    metrics={}
    rows=pd.DataFrame(dict(PassengerId=people.PassengerId[~known],order=g[~known],size=size[~known],truth=y[~known]))
    test_calls={}
    release_names={'nine':'proposal_n24_selected_dissent','111':'proposal_n26_champion_fare_six',
                   '161':'proposal_n26_joint_fare_foundations'}
    for name,v in versions.items():
        rows[name+'_low']=v[0,~known]
        rows[name+'_high']=v[1,~known]
        directory=ART/release_names[name]
        assert dig(directory/'submission.csv')==json.loads((ROOT/'reports'/release_names[name]/'manifest.json').read_text())['sha256']
        frame=pd.read_csv(directory/'submission.csv')
        np.testing.assert_array_equal(test.PassengerId,frame.PassengerId)
        test_calls[name]=frame.Transported.to_numpy(bool)
    for old,new in [('nine','111'),('nine','161'),('111','161')]:
        d=(versions[new]==y).astype(int)-(versions[old]==y).astype(int)
        assert np.array_equal(d[0,known],d[1,known])
        name=f'{old}_to_{new}'
        for sign,j in [('low',0),('high',1)]:
            rows[name+'_'+sign]=d[j,~known]
        metrics[name]={scope:bounds(d,m) for scope,m in [('unknown',~known),('known',known),('all',np.ones(len(y),bool)),
                           ('fresh_known',fresh&known),('fresh_all',fresh)]}
        metrics[name]['size_transfer']=size_audit(d,g,size,fresh,known,ts)
        changes=test_calls[old]!=test_calls[new]
        for category,cell in metrics[name]['size_transfer']['size_cells'].items():
            mask=ts>=4 if category=='4+' else ts==int(category)
            cell['test_changed']=int((mask&changes).sum())
    # Demonstrate the range for a broad, frozen challenger whose predictions
    # are independent of the uncertain baseline decision.
    with np.load(ROOT/'artifacts/nov23_full_roster_booking_validation/mitra96.npz') as z:
        for n,val in [('passenger',people.PassengerId),('fold',fold),('y',y)]:
            np.testing.assert_array_equal(z[n],val)
        fixed=z['p']>=.5
    paid=people[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).any(axis=1).to_numpy()
    sensitivity={}
    for name,selection in [('all',np.ones(len(y),bool)),('positive_bill',paid),('no_positive_bill',~paid)]:
        base=versions['161']
        proposed=np.where(selection,fixed,base)
        d=(proposed==y).astype(int)-(base==y).astype(int)
        sensitivity[name]={scope:bounds(d,m) for scope,m in [('known',known),('unknown',~known),('full',np.ones(len(y),bool))]}
    assert len(rows)==161
    assert metrics['nine_to_111']['known']['net_bounds']==[67,67]
    assert metrics['111_to_161']['known']['net_bounds']==[13,13]
    assert metrics['nine_to_111']['fresh_known']['net_bounds']==[20,20]
    assert metrics['111_to_161']['fresh_known']['net_bounds']==[-1,-1]
    path=ROOT/'artifacts/nov26_clean_fare_unknown_branches/paired_rows.csv'
    path.parent.mkdir(exist_ok=True)
    rows.to_csv(path,index=False)
    report=dict(unknown_rows=len(rows),unknown_bookings=int(rows.order.nunique()),
                broad_fixed_Mitra_sensitivity_vs_161=sensitivity,
                anchor_identified_in_unknown={k:int((v[0,~known]==v[1,~known]).sum()) for k,v in versions.items()},
                absolute_correct={k:bounds((v==y).astype(int),np.ones(len(y),bool))['net_bounds'] for k,v in versions.items()},
                pairs=metrics,unknown_row_cache=str(path.relative_to(ROOT)),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                note='Rowwise extrema allow independently feasible prior directions for unresolved rows and therefore give conservative population bounds if extra cross-passenger constraints exist. The four-size standardization only reports strata actually represented by fresh orders; the binary standardization extrapolates all parties from a small number of fresh multi-passenger orders. Counts use whole-order sizes in each roster.')
    out=ROOT/'reports/nov23_matched_learning_curve/clean_fare_unknown_bounds.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['anchor_identified_in_unknown'],report['absolute_correct'])
    for pair,m in metrics.items():
        print(pair,m['unknown'],m['fresh_known'],m['size_transfer'])


if __name__=='__main__':
    main()
