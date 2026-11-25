"""Estimate local passenger age mixtures from other bookings on the same route and deck-side."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from exaone_validation_data import compact_features
from frozen_queue_shared_errors_n26 import paired,ROOT,FEES
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker

BANDS=[6,13,18,26,41,61]
OUT=ROOT/'artifacts/nov26_berth_age_distribution'


def local_ages(people,groups):
    age=people.Age.to_numpy()
    band=np.digitize(age,BANDS)
    bins=len(BANDS)+1
    labels=[f'agebin_{j}' for j in range(bins)]
    obs=pd.DataFrame(np.eye(bins)[band],columns=labels)
    obs.loc[~np.isfinite(age),labels]=np.nan
    cabin=people.Cabin.str.split('/',expand=True)
    pos=pd.to_numeric(cabin[1]).to_numpy()
    route=(people.HomePlanet.fillna('?')+'_'+people.Destination.fillna('?')+'_'+cabin[0].fillna('?')+cabin[2].fillna('?')).to_numpy()
    obs=obs.assign(route=route,booking=groups,position=pos)
    chance=np.full((len(age),3,bins),np.nan)
    rng=np.random.default_rng(16273)
    for t,q in GroupKFold(4,shuffle=True,random_state=7518).split(age,groups=groups):
        assert not np.isin(groups[t],groups[q]).any()
        donor=obs.iloc[t].dropna(subset=['position',labels[0]]).groupby(['route','booking'],as_index=False)[['position',*labels]].mean()
        for key,bank in donor.groupby('route'):
            ask=q[(route[q]==key)&np.isfinite(pos[q])]
            if not len(ask):continue
            positions=bank.position.to_numpy()
            votes=bank[labels].to_numpy()
            prior=(votes.sum(0)+1)/(len(bank)+bins)
            fake=votes[rng.permutation(len(bank))]
            for ix in np.array_split(ask,max(1,len(ask)//128)):
                gaps=np.abs(pos[ix,None]-positions)
                # Adaptive span covers 24 distinct bookings even in sparsely occupied lanes.
                radius=np.partition(gaps,min(23,len(bank)-1),axis=1)[:,min(23,len(bank)-1)].clip(2)
                weights=np.exp(-gaps/radius[:,None])
                chance[ix,0]=prior
                for j,v in enumerate([votes,fake],1):
                    chance[ix,j]=(weights@v+12*prior)/(weights.sum(1)[:,None]+12)
    use=np.isfinite(chance[:,0,0])&np.isfinite(age)
    np.testing.assert_allclose(chance[use].sum(2),1.,atol=1e-10)
    likelihood=chance[np.arange(len(age))[:,None],np.arange(3),band[:,None]]
    diag=dict(scored=int(use.sum()),age_log_loss=(-np.log(likelihood[use]).mean(0)).tolist())
    features=[]
    for j in range(3):
        density=chance[:,j]
        # The within-route rank and mass of the passenger's own band contextualize recorded age.
        mass=likelihood[:,j].copy()
        rank=np.nansum(density*(np.arange(bins)<band[:,None]),axis=1)+mass/2
        mass[~use]=np.nan
        rank[~use]=np.nan
        features.append(np.column_stack([density,mass,rank]))
    return features,diag


def run():
    raw,y,g,fold,valid,versions=paired()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    roster=pd.concat([raw,test],ignore_index=True)
    group=roster.PassengerId.str[:4].to_numpy()
    features,diag=local_ages(roster,group)
    x=compact_features(raw)
    x['party_size']=raw.PassengerId.groupby(g).transform('size').to_numpy()
    bank=[x]+[pd.concat([x,pd.DataFrame(val[:len(y)],columns=[f'local_bin_{c}' for c in range(val.shape[1])])],axis=1) for val in features]
    p=np.zeros((len(raw),4))
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        for j,fields in enumerate(bank):
            p[q,j]=fit_model(fields.iloc[t],y[t]).predict_proba(fields.iloc[q])[:,1]
        print(k,flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',ids=raw.PassengerId.to_numpy(str),fold=fold,prob=p,y=y)
    prior=versions[:,-1]
    wrong=prior!=y
    common=(versions==prior[:,None]).all(1)
    sleepers=raw.CryoSleep.eq(True).to_numpy()&~raw[FEES].gt(0).any(axis=1).to_numpy()&prior
    scopes={'all':valid,'all_queued_agree':valid&common,'sleeper_false_alarm_cohort':valid&common&sleepers,
            'other_unanimous':valid&common&~sleepers,'solo_unanimous':valid&common&x.party_size.eq(1).to_numpy()}
    assert (scopes['all_queued_agree']&wrong).sum()==1181 and (scopes['sleeper_false_alarm_cohort']&wrong).sum()==388
    edits=(p>=.5)!=prior[:,None]
    actions={**dict(zip(['own','route_prior','nearby','permuted'],edits.T,strict=True)),
             'nearby_extra':edits[:,2]&~edits[:,0], 'permuted_extra':edits[:,3]&~edits[:,0],
             'nearby_exclusive':edits[:,2]&~edits[:,0]&~edits[:,1]&~edits[:,3]}
    report={n:{s:ticker(m,wrong,act) for s,m in scopes.items()} for n,act in actions.items()}
    print(report['nearby_extra'],report['permuted_extra'],diag,flush=True)
    report['age_diagnostic']=diag
    (ROOT/'reports/nov23_matched_learning_curve/berth_age_distribution.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
