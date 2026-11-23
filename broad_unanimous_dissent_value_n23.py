"""Measure the yield of stable dissent on passengers whose earlier decisions were unanimous."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS, ROOT, load
from broad_local_profile_fit_n23 import REPORT
from broad_matched_model_changes_n23 import read_bank
from broad_partition_evidence_n23 import ART,SEEDS,write

SOURCES={'ordinary_neighbors':('local_profile_fit','ordinary'),
         'balanced_neighbors':('local_profile_fit','balanced'),
         'local_slopes':('local_profile_fit','local_slopes'),
         'relative_radii':('class_hulls','radii'),
         'centroids':('class_hulls','centroids'),
         'hulls':('class_hulls','hulls')}


def audit(selected,errors,confidence,decision,cohort,groups):
    # Compare flagged passengers with every member of their prior-direction and risk stratum.
    ids=np.flatnonzero(cohort)
    strata=np.zeros(len(ids),int)
    for direction in np.unique(decision[ids]):
        local=np.flatnonzero(decision[ids]==direction)
        edges=np.unique(np.quantile(confidence[ids[local]],np.arange(1,10)/10))
        strata[local]=10*int(direction)+np.searchsorted(edges,confidence[ids[local]])
    risk=np.bincount(strata,weights=errors[ids])/np.maximum(1,np.bincount(strata))
    flagged=selected[ids]
    rescued=int(errors[ids][flagged].sum())
    lost=int(flagged.sum()-rescued)
    expected=float(risk[strata[flagged]].sum())
    return dict(fixed=rescued,harmed=lost,precision=(rescued/(rescued+lost) if rescued+lost else None),
                expected_errors_at_same_confidence_and_direction=expected,
                affected_bookings=int(np.unique(groups[ids[flagged]]).size))



def example_pairs(frame, row, y, group, vote, cohort, dissent_count):
    # Descriptive comparisons: outcomes identify cases and controls only after the
    # predictions have been evaluated. No matches from this audit train a model.
    fares=frame[BILLS]
    dominant=fares.fillna(-1).idxmax(axis=1).where(fares.gt(0).any(axis=1),'no_positive')
    cat=frame[['HomePlanet','CryoSleep','Destination','Deck','Side']].copy()
    cat['dominant']=dominant
    code=cat.groupby(list(cat.columns),dropna=False).ngroup().to_numpy()
    continuous=frame[['Age','CabinNumber',*BILLS]].rank(pct=True).fillna(-1).to_numpy()
    ids=pd.read_csv(ROOT/'data/raw/train.csv',usecols=['PassengerId']).PassengerId.to_numpy()[row]
    def describe(i):
        vals=frame.iloc[i]
        return dict(passenger_id=str(ids[i]), booking=int(group[i]), home=str(vals.HomePlanet),
                    sleep=str(vals.CryoSleep), destination=str(vals.Destination), deck=str(vals.Deck), side=str(vals.Side),
                    age=None if pd.isna(vals.Age) else float(vals.Age),
                    cabin_number=None if pd.isna(vals.CabinNumber) else int(vals.CabinNumber),
                    bills={k:None if pd.isna(vals[k]) else float(vals[k]) for k in BILLS},
                    historical_prediction=bool(vote[i]), transported=bool(y[i]), stable_contrary_models=int(dissent_count[i]))
    result={}
    for name,eligible in cohort.items():
        if name=='all':continue
        wrong=np.flatnonzero(eligible&(vote!=y))
        correct=np.flatnonzero(eligible&(vote==y))
        matches=[]
        for i in wrong:
            candidates=correct[(code[correct]==code[i])&(group[correct]!=group[i])]
            if len(candidates):
                gap=np.linalg.norm(continuous[candidates]-continuous[i],axis=1)
                j=gap.argmin()
                matches.append((float(gap[j]),int(i),int(candidates[j])))
        if matches:
            distance,i,j=sorted(matches)[len(matches)//2]
            result[name]=dict(cases=len(wrong),matched=len(matches),distinct_controls=len(set(a[2] for a in matches)),
                              representative_distance=distance,case=describe(i),correct=describe(j))
    return result


def main():
    row,y,frame,g,_=load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        vote=z['votes'][0,:,0]
        agrees=(z['votes']==vote[None,:,None]).all((0,2))
    old_conf,new={},{}
    for seed in map(str,SEEDS):
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            fold=z['fold']
            p=read_bank(row,y,fold,seed).reshape(len(y),-1)
            props=np.column_stack([z[k] for k in ('sleep_conditional','sleep_direct','basket_conditional','basket_direct')])
            agrees &= ((np.c_[p,props]>=.5)==vote[:,None]).all(1)
            old_conf[seed]=np.abs(logit(z['base'].clip(.002,.998)))
        for name,(directory,key) in SOURCES.items():
            with np.load(ART/f'nov23_{directory}/{seed}.npz') as z:
                np.testing.assert_array_equal(row,z['row'])
                np.testing.assert_array_equal(fold,z['fold'])
                new.setdefault(name,[]).append(z[key])
    errors=vote!=y
    assert (errors&agrees).sum()==372 and (~errors&agrees).sum()==3867
    confidence=np.mean(list(old_conf.values()),axis=0)
    risk=frame[['VRDeck','Spa','RoomService']].gt(0).any(axis=1).to_numpy()
    zero=~frame[['VRDeck','Spa','RoomService','FoodCourt','ShoppingMall']].gt(0).any(axis=1).to_numpy()
    groups={'all':agrees,'paid_predicted_stay':agrees&risk&~vote,'no_positive_predicted_transport':agrees&zero&vote}
    dissent={}
    for k,p in new.items():
        dissent[k]=((np.stack(p)>=.5)!=vote[None,:]).all(0)
    n=np.stack(list(dissent.values())).sum(0)
    for threshold in (1,2,4,6):
        dissent['at_least_'+str(threshold)]=n>=threshold
    out={}
    for key,subset in groups.items():
        out[key]=dict(passengers=int(subset.sum()),errors=int((subset&errors).sum()),
                      confidence_auc=float(roc_auc_score(errors[subset],-confidence[subset])),
                      rules={k:audit(flag,errors,confidence,vote,subset,g) for k,flag in dissent.items()})
    out['matched_examples']=example_pairs(frame,row,y,g,vote,groups,n)
    write(REPORT/'unanimous_dissent.json',out)
    print(out['all'],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
