"""Count contradictory training labels within exactly identical observable feature profiles."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from repeated_archive_misses_n26 import ROOT,cohort
from price_implied_deck_n26 import FEES


def run():
    frame,y,groups,fold,_,_,_,seven,versions=inputs()
    n=len(y); data=frame.iloc[:n].copy(); parent=versions['random'][:n,0]
    _,_,known,_=cohort(data,y,fold)
    common=known&(versions['random'][:n]==parent[:,None]).all(1)&((seven[:n]>=.5)==parent[:,None]).all(1)
    fields=['HomePlanet','CryoSleep','Destination','Age','VIP',*FEES]
    cabin=data.Cabin.str.split('/',expand=True)
    data['Deck']=cabin[0];data['Side']=cabin[2]
    data['GroupSize']=data.groupby(groups).PassengerId.transform('size')
    data['Position']=data.PassengerId.str.split('_').str[-1]
    data['CabinOccupancy']=data.Cabin.map(frame.Cabin.value_counts())
    scopes={'all':np.ones(n,bool),'known':known,'shared':common}
    designs={'all_fields_with_booking_context':[*fields,'Cabin','GroupSize','Position','CabinOccupancy'],
             'all_personal_fields_and_full_cabin':[*fields,'Cabin'],
             'all_personal_fields_deck_side':[*fields,'Deck','Side'],
             'personal_fields_only':fields}
    report={}
    for name,columns in designs.items():
        codes=data.groupby(columns,dropna=False,sort=False).ngroup().to_numpy()
        assert np.isfinite(codes).all()
        sizes=np.bincount(codes)
        pos=np.bincount(codes,weights=y).astype(int)
        mixed=(pos>0)&(pos<sizes)
        sub={}
        for key,mask in scopes.items():
            codeset=codes[mask];counts=np.bincount(codeset,minlength=len(sizes))
            positives=np.bincount(codeset,weights=y[mask],minlength=len(sizes)).astype(int)
            sub[key]=dict(rows=int(mask.sum()),replayed_errors=int(((parent!=y)&mask&known).sum()),
                rows_in_conflict=int(mixed[codeset].sum()),
                errors_in_conflict=int((mixed[codes]&(parent!=y)&mask&known).sum()),
                minimum_errors_for_this_feature_set=int(np.minimum(positives,counts-positives).sum()),
                rows_with_exact_twins=int((sizes[codeset]>1).sum()))
        report[name]=sub
    (ROOT/'reports/nov23_matched_learning_curve/identical_manifest_errors.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    with threadpool_limits(1):run()
