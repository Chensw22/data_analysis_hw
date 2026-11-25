"""Measure label collisions and votes from other bookings at successive manifest resolutions."""
import json
import numpy as np
import pandas as pd
from full_roster_booking_validation_n23 import ROOT,data

REPORT=ROOT/'reports/nov23_matched_learning_curve/manifest_twin_audit.json'
BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']


def signatures(raw,g):
    cabin=raw.Cabin.str.split('/',expand=True)
    exact=raw[['HomePlanet','Destination','CryoSleep','VIP']].copy()
    exact['deck'],exact['side']=cabin[0],cabin[2]
    exact['party']=raw.PassengerId.groupby(g).transform('size')
    exact['age']=raw.Age
    exact['number']=pd.to_numeric(cabin[1])
    exact[BILLS]=raw[BILLS]
    versions={'exact_without_identifiers':exact}
    for mode in ('neighborhood','behavior','service_pattern'):
        frame=exact.copy()
        frame['age']=np.floor(frame.age/5)
        if mode=='neighborhood':
            frame['number']=np.floor(frame.number/50)
        else:
            frame=frame.drop(columns='number')
        for col in BILLS:
            v=frame[col]
            if mode=='service_pattern':
                frame[col]=np.where(v.isna(),-1,(v>0).astype(int))
            else:
                # Missing, zero and distinct positive spend bands retain different codes.
                frame[col]=pd.cut(v,[-1,0,50,250,750,1500,3000,np.inf],labels=False)
        versions[mode]=frame
    return versions


def run():
    raw,y,g,f=data()
    audit=pd.read_csv(ROOT/'artifacts/nov25_within_bin_bill_precision/passenger_error_audit.csv',dtype={'PassengerId':str})
    np.testing.assert_array_equal(audit.PassengerId,raw.PassengerId)
    np.testing.assert_array_equal(audit.booking_fold,f)
    np.testing.assert_array_equal(audit.transported,y)
    votes=audit.original_votes_true.to_numpy()
    consensus=(votes==68)|(votes==0)
    solo=~pd.Series(g).duplicated(keep=False).to_numpy()
    hardest=(votes==0)&raw.CryoSleep.eq(False).to_numpy()&solo
    ref=audit.reference_nine
    known=ref.notna().to_numpy()
    nine=ref.astype('boolean').fillna(False).to_numpy(bool)
    scopes={'all_nine':known,'unanimous':known&consensus,'awake_solo_predicted_false':known&hardest,
            'unseen_orders':known&audit.new_order.to_numpy(),'all_143':hardest&y}
    r={}
    for mode,frame in signatures(raw,g).items():
        # Assign a distinct ID to each exact tuple, including its pattern of missing fields.
        cell=frame.groupby(frame.columns.tolist(),dropna=False,sort=False).ngroup().to_numpy()
        assert (cell>=0).all()
        width=int(cell.max()+1)
        total=np.bincount(cell,minlength=width)
        positive=np.bincount(cell,weights=y,minlength=width).astype(int)
        mixed=(positive>0)&(positive<total)
        minimum_errors=int(np.minimum(positive,total-positive).sum())
        support=np.zeros(len(y),int)
        yes=np.zeros(len(y),int)
        for k in np.unique(f):
            t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
            assert not np.isin(g[t],g[q]).any()
            support[q]=np.bincount(cell[t],minlength=width)[cell[q]]
            yes[q]=np.bincount(cell[t],weights=y[t],minlength=width).astype(int)[cell[q]]
        # All votes come from other bookings. Retain the original decision on tied donor vote.
        change=(support>=3)&np.where(nine,2*yes<support,2*yes>support)
        summary={}
        for name,m in scopes.items():
            summary[name]=dict(rows=int(m.sum()),with_any_twin=int((m&(support>0)).sum()),
                mixed_donor_outcomes=int((m&(yes>0)&(yes<support)).sum()),
                at_least_three=int((m&(support>=3)).sum()),
                majority_transported=int((m&(support>=3)&(2*yes>support)).sum()),
                fixed=int((m&known&change&(nine!=y)).sum()),harmed=int((m&known&change&(nine==y)).sum()))
        r[mode]=dict(unique_signatures=len(total),passengers_in_mixed_cells=int(total[mixed].sum()),
            minimum_in_sample_errors_using_only_this_signature=minimum_errors,scopes=summary)
        print(mode,minimum_errors,summary['all_143'],summary['all_nine']['fixed'],summary['all_nine']['harmed'])
    assert r['exact_without_identifiers']['scopes']['all_143']['rows']==143
    REPORT.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    run()
