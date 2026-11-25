"""Count conflicting observable passenger profiles and transfer outcomes from distinct matching bookings."""
import json
import numpy as np
import pandas as pd
from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import archives,projection
from full_roster_booking_validation_n23 import ROOT
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker


def profile_keys(raw,groups):
    cabin=raw.Cabin.str.split('/',expand=True)
    bill=raw[FEES]
    table=raw[['HomePlanet','CryoSleep','Destination','VIP','Age',*FEES]].copy()
    table[['deck','berth','side']]=cabin
    table['party']=raw.PassengerId.groupby(groups).transform('size')
    table['paid']=bill.gt(0).any(axis=1)
    table['age_band']=np.floor(raw.Age/8)
    table['size_band']=table.party.clip(upper=4)
    table['n_services']=bill.gt(0).sum(axis=1)
    table['dominant']=np.where(table.paid,bill.fillna(0).idxmax(axis=1),'none')
    for fee in FEES:
        table[fee+'_band']=np.where(bill[fee].isna(),-1,np.ceil(np.log2(1+bill[fee].fillna(0))))
    get=lambda cols: table.groupby(cols,sort=False,dropna=False).ngroup().to_numpy(int)
    base=['CryoSleep','paid','HomePlanet']
    loc=['deck','side','Destination','VIP']
    spend=['dominant','n_services',*[n+'_band' for n in FEES]]
    life=['age_band','size_band']
    exact=['Age','party',*FEES]
    result={}
    for name,additions in [('registration',[base,loc,life,spend,exact,['berth']]),
                           ('services',[['paid','dominant','n_services'],[n+'_band' for n in FEES],base+life,loc,exact,['berth']])]:
        columns=[]
        keys=[]
        for more in additions:
            columns=list(dict.fromkeys([*columns,*more]))
            keys.append(get(columns))
        result[name]=np.stack(keys)
    return result


def equivalent_outcomes(keys,rows,groups,truth,wrong):
    frame=pd.DataFrame(dict(profile=keys[rows],group=groups[rows],label=truth[rows].astype(int)))
    counts=frame.groupby('profile').label.agg(['sum','size'])
    counts['no']=counts['size']-counts['sum']
    cases=frame.join(counts,on='profile')
    both=(cases['sum']>0)&(cases.no>0)
    # Count opposite labels after subtracting all members of the focal booking.
    peers=frame.groupby(['profile','group']).label.agg(['sum','size'])
    peers.columns=['own_true','own_size']
    cases=cases.join(peers,on=['profile','group'])
    opposite=np.where(cases.label,cases.no-(cases.own_size-cases.own_true),cases['sum']-cases.own_true)>0
    return dict(rows=len(rows),profiles=len(counts),mixed_label_rows=int(both.sum()),
                opposite_in_another_booking=int(opposite.sum()),errors_with_opposite_elsewhere=int((opposite&wrong[rows]).sum()),
                min_errors_for_signature=int(np.minimum(counts['sum'],counts.no).sum()))


def lookup(keys,t,q,groups,y):
    assert not np.isin(groups[t],groups[q]).any()
    tables=[]
    for level in keys:
        bank=pd.DataFrame(dict(key=level[t],group=groups[t],label=y[t].astype(float)))
        averaged=bank.groupby(['key','group']).label.mean().reset_index()
        tables.append({k:(part.group.to_numpy(),part.label.to_numpy()) for k,part in averaged.groupby('key')})
    prior=float(y[t].mean())
    scores=[]
    # Within a given level all tied bookings get the same weight.
    for i in q:
        seen=set()
        total=positive=0.
        for level,bank in reversed(list(zip(keys,tables,strict=True))):
            entry=bank.get(level[i])
            if entry is None:
                continue
            ids,labels=entry
            take=np.fromiter((order not in seen for order in ids),dtype=bool,count=len(ids))
            available=int(take.sum())
            if not available:
                continue
            use=min(24-total,available)
            positive+=use*labels[take].mean()
            total+=use
            if total>=24:
                break
            seen.update(ids[take])
        scores.append((positive+4*prior)/(total+4))
    return np.array(scores)


def run():
    raw,y,g,fold,views,original=projection(include_full=True)
    nine,known,_,_=champions()
    keys=profile_keys(raw,g)
    np.testing.assert_array_equal(keys['registration'][-1],keys['services'][-1])
    extra,_=archives(raw,y,g,fold,True)
    outer=np.column_stack([np.mean([extra[n] for n in ('Mitra1_frozen','Mitra2_frozen','Mitra2_adapted')],axis=0),
                           extra['TabDPT'],extra['LimiX2'],extra['EXAONE_grouped']])>=.5
    predictions={}
    for name,levels in keys.items():
        score=np.full(len(y),np.nan)
        other=np.full(len(y),np.nan)
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            score[q]=lookup(levels,t,q,g,y)
            # Hold fixed the three coarse registration fields or service fields and remove finer matching.
            other[q]=lookup(levels[:1],t,q,g,y)
        predictions[name]=score
        predictions[name+'_coarse']=other
    whole=equivalent_outcomes(keys['registration'][-1],np.arange(len(y)),g,y,np.zeros(len(y),bool))
    del whole['errors_with_opposite_elsewhere']
    report={'whole_labeled_roster_exact':whole}
    for tag in ('full','bookingmix','historical_to_new'):
        valid,parent=views[tag]
        assert known[valid].all()
        joint=valid&(parent==nine)
        wrong=parent!=y
        rows=np.flatnonzero(joint)
        summary={}
        for name,levels in keys.items():
            summary[name]=[equivalent_outcomes(level,rows,g,y,wrong) for level in levels]
        actions={n:(v>=.5)!=parent for n,v in predictions.items()}
        actions['both_views']=actions['registration']&actions['services']
        actions['new_in_both_views']=actions['both_views']&~actions['registration_coarse']&~actions['services_coarse']
        signatures=(original[:,:18]!=parent[:,None]).sum(1)>=10
        outside=(outer!=parent[:,None]).sum(1)>=3
        actions['new_with_family_majority']=actions['new_in_both_views']&signatures
        actions['new_with_foundation_majority']=actions['new_in_both_views']&outside
        actions['new_with_both_majorities']=actions['new_in_both_views']&signatures&outside
        r={n:ticker(joint,wrong,action) for n,action in actions.items()}
        report[tag]=dict(collisions=summary,transferred=r)
        print(tag,summary['registration'][-1],r['both_views'],r['new_in_both_views'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/shared_error_profile_collisions.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
