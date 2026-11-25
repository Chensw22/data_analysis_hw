"""Give raw-valued TabICL the same four missing sources as the joint manifest model."""
import hashlib
import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov25_precise_source_views'
GROUPS={'registration':['HomePlanet','Destination','VIP','Age'],'sleep':['CryoSleep'],
        'cabin':['Deck','Side','CabinNumber'],'bills':['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']}


def run():
    import torch
    from tabicl_routed_context_n23 import WEIGHT,DIGEST,infer
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    raw,y,g,fold=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    x=compact_features(pd.concat([raw,test],ignore_index=True))
    assert set(x)=={col for fields in GROUPS.values() for col in fields}
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        for name,fields in GROUPS.items():
            frame=x.drop(columns=fields)
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                p=infer(frame,y,t,q,891+k)
                assert len(p)==len(q) and ((p>=0)&(p<=1)).all()
                with path.with_suffix('.tmp').open('wb') as out:
                    np.savez_compressed(out,p=p,q=q,t=t,labels=y[t],fields=np.asarray(frame.columns,dtype=str),
                                        passenger=raw.PassengerId.iloc[q].to_numpy(str),checkpoint=DIGEST)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('q',q),('t',t),('labels',y[t]),('fields',frame.columns),('passenger',raw.PassengerId.iloc[q]),
                                ('checkpoint',DIGEST)]:
                    np.testing.assert_array_equal(z[key],val)
            print(k,name,flush=True)


if __name__=='__main__':
    run()
