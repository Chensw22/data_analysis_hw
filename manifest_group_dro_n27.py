"""Train passenger trees with an entropic worst-group loss over observable manifest strata."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit, softmax
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from export_directional_foundation_n27 import NAME as CHAMP
from full_roster_two_view_reliability_n24 import design
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from six_way_against_scored_nine_n26 import test_decisions

OUT=ART/'nov27_manifest_group_dro'
REPORT=ROOT/'reports/nov23_matched_learning_curve/manifest_group_dro.json'
FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
TEMPERATURE=.16


def strata(raw,groups,t,named=False):
    bills=raw[FEES].fillna(0)
    total=bills.sum(axis=1)
    favorable=bills[['FoodCourt','ShoppingMall']].sum(axis=1)
    service=np.select([total.eq(0)&raw.CryoSleep.eq(True),total.eq(0),favorable>total/2],['sleep','zero','retail'],default='other_paid')
    size=raw.groupby(groups).PassengerId.transform('size').clip(upper=3)
    small=raw.HomePlanet.fillna('?')+'|'+service
    all_cells=small+'|'+size.astype(str)
    labels=all_cells.iloc[t]
    # Sparse cells are pooled into their larger registration/spending parent, then service group.
    lbl=labels.where(labels.map(labels.value_counts())>=45,small.iloc[t])
    lbl=lbl.where(lbl.map(lbl.value_counts())>=45,service[t])
    lbl=lbl.where(lbl.map(lbl.value_counts())>=45,'rare')
    return lbl.to_numpy(str) if named else np.unique(lbl.to_numpy(str),return_inverse=True)[1]


def derivatives(z,y,env,mode):
    n=len(y)
    prob=expit(z)
    grad=prob-y
    hess=prob*(1-prob)
    if mode=='ordinary':return grad,hess.clip(.0001)
    counts=np.bincount(env)
    losses=np.logaddexp(0,z)-y*z
    avg=np.bincount(env,weights=losses)/counts
    # Starting mass is the empirical population share. Exponentially tilt it towards high-loss groups.
    mass=softmax(np.log(counts/n)+avg/TEMPERATURE)
    weight=n*mass[env]/counts[env]
    diagonal=hess+(1-mass[env])*grad**2/(TEMPERATURE*counts[env])
    return weight*grad,(weight*diagonal).clip(.0001)


def fit(x,y,groups,env,t,q):
    assert not np.isin(groups[t],groups[q]).any()
    dataset=lgb.Dataset(x[t],label=y[t].astype(float),free_raw_data=False)
    outputs={}
    for mode in ['ordinary','observable','shuffled']:
        group=env if mode!='shuffled' else np.random.default_rng(554).permutation(env)
        def objective(z,dataset):
            return derivatives(z,y[t],group,mode)
        params=dict(objective=objective,verbosity=-1,num_threads=2,force_col_wise=True,seed=7461,
                    num_leaves=22,learning_rate=.035,min_data_in_leaf=32,lambda_l2=5,
                    feature_pre_filter=False)
        model=lgb.train(params,dataset,num_boost_round=280)
        outputs[mode]=expit(model.predict(x[q],raw_score=True,num_threads=2))
    return outputs


def run():
    raw,y,g,f=data()
    n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    roster=pd.concat([raw,test],ignore_index=True)
    ids=roster.PassengerId.to_numpy(str)
    groups=roster.PassengerId.str.split('_').str[0].to_numpy()
    x=np.column_stack(design(roster,groups))
    _,blind,known,fresh=cohort(raw,y,f)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['fold'],f)
        panel=z['scores'][:,-3:].clip(.001,.999)
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        first=z['champion111'].astype(bool)
    # All splits evaluate against the same held-out representation of the current release.
    med=np.median(panel[:n],axis=1)
    risk=np.where(first,1-med,med)
    part=partition(g,'directional_foundation',4)
    base=first.copy()
    for j in range(8):
        for m in range(4):
            q=np.flatnonzero((f==j)&(part==m))
            t=np.flatnonzero(known&(f==j)&(part!=m))
            cut=calibrate(risk,first,y,t,1.)
            base[q]^=risk[q]>=cut[first[q].astype(int)]
    champion=test_decisions(CHAMP,pd.Series(ids[n:]))[0]
    anchor=np.r_[base,champion]
    native=np.where(anchor,1.,-1.)*abs(logit(panel.mean(1)))
    other=np.full(n,-1)
    for j,orders in enumerate(np.array_split(np.unique(g),8)):other[np.isin(g,orders)]=j
    assert (other>=0).all()
    OUT.mkdir(exist_ok=True)
    report={};cache={}
    masks=dict(all=known,fresh=fresh&known,common31=blind,negative=known&~base,positive=known&base)
    for tag,fold in [('random',f),('ordered',other)]:
        scores={k:np.full(len(ids),np.nan) for k in ['ordinary','observable','shuffled']}
        for j in range(9):
            t=np.flatnonzero(fold!=j) if j<8 else np.arange(n)
            q=np.flatnonzero(fold==j) if j<8 else np.arange(n,len(ids))
            path=OUT/(f'{tag}_{j}.npz' if j<8 else 'test.npz')
            if not path.exists():
                env=strata(roster,groups,t)
                pred=fit(x,y,groups,env,t,q)
                temp=path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,t=t,q=q,ids=ids[q],labels=y[t],sizes=np.bincount(env),**pred)
                temp.replace(path)
            with np.load(path) as z:
                for k,v in [('t',t),('q',q),('ids',ids[q]),('labels',y[t])]:np.testing.assert_array_equal(z[k],v)
                for k in scores:scores[k][q]=z[k]
        robust=logit(scores['observable'].clip(.001,.999))
        for control in ('ordinary','shuffled'):
            increment=robust-logit(scores[control].clip(.001,.999))
            scores['over_'+control]=expit(native+increment)
        acts={k:(v>=.5)!=anchor for k,v in scores.items()}
        acts['both_increments']=acts['over_ordinary']&acts['over_shuffled']
        acts['supported']=acts['both_increments']&acts['observable']
        acts['only_observable']=acts['observable']&~acts['ordinary']&~acts['shuffled']
        hard=strata(roster,groups,np.arange(n))
        result={}
        for k,flag in acts.items():
            d=flag[:n]*np.where(base!=y,1,-1)
            result[k]=dict(test_changes=int(flag[n:].sum()),folds=[int(d[(fold==a)&known].sum()) for a in range(8)],
                scopes={s:count(flag[:n],m,base!=y) for s,m in masks.items()})
            cache[tag+'_'+k]=flag
        report[tag]=dict(rules=result,objectives={k:dict(accuracy=float(((v[:n]>=.5)==y).mean()),
                    worst_group_loss=float(max(np.mean(np.where(y[m],-np.log(v[:n][m]),-np.log1p(-v[:n][m]))) for m in [hard==i for i in np.unique(hard)])))
                    for k,v in scores.items() if k in ('ordinary','shuffled','observable')})
        labels=strata(roster,groups,np.arange(n),named=True)
        per_group={}
        for name in np.unique(labels):
            m=labels==name
            per_group[name]=dict(rows=int(m.sum()),base_errors=int((m&known&(base!=y)).sum()),
                blind_errors=int((m&blind&(base!=y)).sum()),
                logloss={k:float(np.mean(np.where(y[m],-np.log(v[:n][m]),-np.log1p(-v[:n][m]))))
                         for k,v in scores.items() if k in ('ordinary','observable','shuffled')})
        report[tag]['groups']=per_group
    common={}
    for k in acts:
        r=cache['random_'+k];o=cache['ordered_'+k]
        np.testing.assert_array_equal(r[n:],o[n:])
        common[k]={s:count(r[:n]&o[:n],m,base!=y) for s,m in masks.items()}
    np.savez_compressed(OUT/'aligned.npz',passenger=ids,anchor=anchor,fold=f,ordered=other,y=y,**cache)
    report['replicated_same_person']=common
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['random']['objectives'],report['ordered']['objectives'],common['supported'])


if __name__=='__main__':
    with threadpool_limits(2):run()
