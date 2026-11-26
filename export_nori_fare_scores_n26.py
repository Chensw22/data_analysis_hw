"""Refit an improved Nori representation on all released training orders for joint evaluation."""
import argparse
import numpy as np
import pandas as pd

from nori_booking_regression_n26 import CACHE,design,packet_dir
from nori_fare_party_n26 import ART,ROOT,SHA,data,fit,predictor


def export(mode):
    raw,y,g,f=data()
    correct=[]
    for k in range(8):
        with np.load(ART/f'{mode}{k}.npz' if mode!='raw' else packet_dir(0)/f'fold{k}.npz') as new,np.load(packet_dir(0)/f'fold{k}.npz') as old:
            q=np.flatnonzero(f==k)
            np.testing.assert_array_equal(q,new['q'])
            np.testing.assert_array_equal(q,old['q'])
            assert str(new['sha' if mode!='raw' else 'checkpoint'])==SHA and str(old['checkpoint'])==SHA
            correct.append(int(((new['p']>=.5)==y[q]).sum()-((old['p']>=.5)==y[q]).sum()))
    if mode!='raw' and (sum(correct)<=0 or sum(n>0 for n in correct)<=4):
        raise ValueError(f'{mode}: no replicated net gain over raw Nori: {correct}')
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    frame=pd.concat([raw,test],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    t=np.arange(len(y));q=np.arange(len(y),len(frame))
    assert not np.isin(groups[t],groups[q]).any() and len(q)==4277
    resources=predictor()
    if mode=='raw':
        torch,regressor=resources
        x=design(frame,groups,t)
        torch.manual_seed(6117)
        np.random.seed(6117)
        model=regressor(model_path=str(CACHE/'nori.pt'),model='nori-6m',device='cpu',
                        inference_config=str(CACHE/'one_member.json'),categorical_columns=[],augmentations=())
        model.fit(x[t],y[t].astype(float))
        p=np.asarray(model.predict(x[q]),float).ravel()
        assert p.shape==(len(q),) and np.isfinite(p).all()
        packet=dict(t=t,q=q,labels=y[t],passenger=frame.PassengerId.iloc[q].to_numpy(str),p=p,sha=SHA)
    else:
        packet=fit(frame,y,groups,t,q,mode,6117,resources)
    np.testing.assert_array_equal(packet['passenger'],test.PassengerId.to_numpy(str))
    np.savez_compressed(ART/f'{mode}_test.npz',**packet,oof_net=correct)
    print(mode,'test probabilities',len(q),'local net per fold',correct,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['raw','own','peer'])
    export(parser.parse_args().mode)
