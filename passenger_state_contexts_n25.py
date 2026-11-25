"""Test full-roster TabICL against contexts based on sleep declaration and recorded service use."""
import hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from awake_context_ranks_n25 import BILLS, OUT as AWAKE
from exaone_validation_data import CATEGORICAL,compact_features
from full_roster_booking_validation_n23 import ROOT,data

OUT=ROOT/'artifacts/nov25_passenger_state_contexts'


def sleep_prior(x,t,q,asleep):
    # Predict sleep status from fields available to every passenger with missing sleep declaration.
    fields=x.drop(columns='CryoSleep')
    model=CatBoostClassifier(iterations=240,depth=4,learning_rate=.07,random_seed=357,thread_count=2,
                             verbose=False,allow_writing_files=False)
    model.fit(fields.iloc[t],asleep[t],cat_features=[c for c in CATEGORICAL if c!='CryoSleep'])
    return model.predict_proba(fields.iloc[q])[:,1]


def run():
    import torch
    from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,fold=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    x=compact_features(pd.concat([raw,test],ignore_index=True)).iloc[:len(y)].copy()
    awake=raw.CryoSleep.eq(False).to_numpy()
    asleep=raw.CryoSleep.eq(True).to_numpy()
    missing=~(awake|asleep)
    paid=raw[list(BILLS)].gt(0).any(axis=1).to_numpy()
    x_awake=x.copy()
    x_sleep=x.copy()
    x_awake.loc[missing,'CryoSleep']='False'
    x_sleep.loc[missing,'CryoSleep']='True'
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        t,q=fold!=k,fold==k
        sleep_t=np.flatnonzero(t&(awake|asleep))
        missing_q=np.flatnonzero(q&missing)
        prior=sleep_prior(x,sleep_t,missing_q,asleep)
        plans=[
            ('global',x,t,q),
            ('asleep',x_sleep,t&asleep,q&(asleep|missing)),
            ('awake_missing',x_awake,t&awake,q&missing),
            ('paid',x,t&paid,q&paid),
            ('no_payment',x,t&~paid,q&~paid)]
        for name,frame,train,query in plans:
            tr,qr=np.flatnonzero(train),np.flatnonzero(query)
            assert not np.isin(g[tr],g[qr]).any()
            path=OUT/f'{k}_{name}.npz'
            if not path.exists():
                p=infer(frame,y,tr,qr,891+k)
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.savez_compressed(stream,t=tr,q=qr,train_y=y[tr],p=p,passenger=raw.PassengerId.iloc[qr].to_numpy(str),
                                        missing_q=missing_q,prior=prior,checkpoint=DIGEST)
                path.with_suffix('.tmp').replace(path)
            with np.load(path) as z:
                for key,val in [('t',tr),('q',qr),('train_y',y[tr]),('passenger',raw.PassengerId.iloc[qr]),
                                ('missing_q',missing_q),('prior',prior),('checkpoint',DIGEST)]:
                    np.testing.assert_array_equal(z[key],val)
            print(k,name,flush=True)


if __name__=='__main__':
    run()
