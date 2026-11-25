"""Look for residual error information on new bookings after matching the 214-edit candidate's decisions and confidence."""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import DEST
from exaone_validation_data import compact_features
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import data
from joint_candidate_coverage_n26 import branch_decisions


def design(raw,prob):
    x=compact_features(raw)
    bills=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    amount=np.log1p(x[bills].fillna(0)).to_numpy()
    cat=pd.get_dummies(x[['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']].astype(str),dtype=float)
    # Cabin number is observable on its own deck. Numeric group IDs and names do not enter.
    fields=np.column_stack([cat,x.Age.fillna(-1),x.CabinNumber.fillna(-1),amount,x[bills].isna(),x[bills].gt(0)])
    odds=np.log(prob.clip(.005,.995)/(1-prob.clip(.005,.995)))
    return {'manifest':fields,'opinions':odds}


def pairs(rows,wrong,cell,margin):
    left=[];right=[]
    for key in np.unique(cell[rows]):
        sub=rows[cell[rows]==key]
        cases=sub[wrong[sub]]
        controls=sub[~wrong[sub]]
        if not len(cases) or not len(controls):
            continue
        # Resolve competition for controls by ordering edges by confidence distance first.
        distance=np.abs(margin[cases,None]-margin[controls])
        a,b=np.where(distance<=.35)
        order=np.argsort(distance[a,b],kind='stable')
        used_a=set();used_b=set()
        for j in order:
            i,k=int(a[j]),int(b[j])
            if i not in used_a and k not in used_b:
                left.append(cases[i]);right.append(controls[k])
                used_a.add(i);used_b.add(k)
    return np.asarray(left,dtype=int),np.asarray(right,dtype=int)


def kernel_map(values,fit_rows,seed):
    mu=values[fit_rows].mean(axis=0)
    scale=values[fit_rows].std(axis=0).clip(.12)
    x=((values-mu)/scale).clip(-5,5)
    rng=np.random.default_rng(seed)
    w=rng.normal(size=(x.shape[1],256))/np.sqrt(x.shape[1])
    b=rng.uniform(0,2*np.pi,size=256)
    return np.cos(x@w+b)


def run():
    raw,y,g,f=data()
    _,_,_,oldrows,names,p,_,native,_,r=panels_and_reference()
    awake=raw.CryoSleep.eq(False).to_numpy()
    all_ids=np.arange(len(y))
    pred=branch_decisions(all_ids,f,p,names,native,*r,awake,'full',891)
    anchor=pred['all_three'][0]
    known=r[0]>=0
    errors=anchor!=y
    fresh=~np.isin(g,g[oldrows])
    size=raw.PassengerId.groupby(g).transform('size').to_numpy()
    sleep=raw.CryoSleep.map({False:0,True:1}).fillna(2).to_numpy(int)
    cell=anchor.astype(int)+2*(size>1)+4*sleep
    margin=np.log(native.clip(.002,.998)/(1-native.clip(.002,.998)))
    # Pick a representative per booking without reference to any observed outcome.
    order=np.random.default_rng(326).permutation(len(y))
    representative=np.zeros(len(y),bool)
    representative[order[~pd.Series(g[order]).duplicated().to_numpy()]]=True
    assert np.unique(g[representative]).size==representative.sum()
    votes=(p>=.5).sum(1)
    cohorts={'all':known,'unanimous_negative':known&(votes==0),
             'awake_solo_negative':known&(votes==0)&awake&(size==1)}
    data_sets=design(raw,p)
    results={}
    for cohort,scope in cohorts.items():
        pieces={}
        for k in range(8):
            eligible=scope&representative&(f==k)
            t,q=all_ids[eligible&~fresh],all_ids[eligible&fresh]
            a,b=pairs(t,errors,cell,margin)
            c,d=pairs(q,errors,cell,margin)
            if not len(a) or not len(c):
                continue
            assert not np.isin(g[t],g[q]).any() and len(np.unique(g[np.r_[a,b,c,d]]))==2*(len(a)+len(c))
            rng=np.random.default_rng(3006+k)
            signs=rng.choice([-1,1],size=(len(a),63))
            train_features={}
            test_features={}
            for i,(name,values) in enumerate(data_sets.items()):
                embedding=kernel_map(values,t,561+i)
                train_features[name]=embedding[a]-embedding[b]
                test_features[name]=embedding[c]-embedding[d]
            train_features['both']=np.column_stack(list(train_features.values()))
            test_features['both']=np.column_stack(list(test_features.values()))
            for name,train in train_features.items():
                test=test_features[name]
                direction=train.mean(0)
                null=train.T@signs/len(a)
                pieces.setdefault(name,[]).append((test@direction,test@null,np.abs(margin[c]-margin[d]),len(a)))
        table={}
        for name,parts in pieces.items():
            diffs,null,gaps=(np.concatenate([v[j] for v in parts]) for j in range(3))
            assert np.isfinite(diffs).all()
            observed=int((diffs>0).sum())
            ties=int((diffs==0).sum())
            null_wins=(null>0).sum(0)
            table[name]=dict(pairs=len(diffs),historical_pairs=sum(v[3] for v in parts),
                ranks_error_higher=observed,ties=ties,median_confidence_gap_log_odds=float(np.median(gaps)),
                one_sided_p=float(binomtest(observed,len(diffs)-ties,p=.5,alternative='greater').pvalue),
                shuffled_wins_median=float(np.median(null_wins)),
                shuffled_wins_range=np.quantile(null_wins,[.025,.975]).tolist())
        results[cohort]=table
        print(cohort,[(key,r['ranks_error_higher'],r['pairs']) for key,r in table.items()])
    DEST.with_name('matched_remaining_error_test.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
