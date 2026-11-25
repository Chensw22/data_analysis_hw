"""Read event probabilities and inner intervals from Nori's 999 predicted quantiles."""
import argparse
import json
import os
import sys
import numpy as np

from nori_booking_regression_n26 import CACHE, OUT, ROOT, SHA, data, design, packet_dir

REPORT = ROOT/'reports/nov23_matched_learning_curve/nori_binary_mass.json'


def extract(context):
    os.environ['MPLCONFIGDIR']=str(CACHE/'mpl')
    os.environ['HF_HUB_OFFLINE']='1'
    sys.path.insert(0,str(CACHE/'lib'))
    from synthefy_nori import NoriRegressor
    import torch
    import hashlib
    assert hashlib.sha256((CACHE/'nori.pt').read_bytes()).hexdigest()==SHA
    torch.set_num_threads(3)
    raw,y,g,f=data()
    for k in range(8):
        folder=packet_dir(context)
        path=folder/f'mass{k}.npz'
        if path.exists():continue
        with np.load(folder/f'fold{k}.npz') as z:
            q,ids,prior=z['q'],z['context'],z['p']
            assert str(z['checkpoint'])==SHA and not np.isin(g[q],g[ids]).any()
            np.testing.assert_array_equal(z['labels'],y[ids])
        x=design(raw,g,ids)
        np.random.seed(6117+k);torch.manual_seed(6117+k)
        m=NoriRegressor(model_path=str(CACHE/'nori.pt'),model='nori-6m',device='cpu',
            inference_config=str(CACHE/'one_member.json'),categorical_columns=[],augmentations=())
        m.fit(x[ids],y[ids].astype(float))
        packet=m.predict(x[q],output_type='full')
        quantiles,taus=packet['quantiles'],packet['taus']
        np.testing.assert_allclose(packet['mean'],prior,rtol=1e-4,atol=1e-4)
        assert quantiles.shape==(len(q),len(taus)) and np.all(np.diff(quantiles,axis=1)>=0)
        mass=np.array([[1-np.interp(level,row,taus) for level in (.25,.5,.75)] for row in quantiles])
        std=quantiles.std(1)
        np.savez_compressed(path,q=q,ids=raw.PassengerId.iloc[q].to_numpy(str),context=ids,labels=y[ids],
                            mass=mass,std=std,median=quantiles[:,len(taus)//2],mean=packet['mean'],checkpoint=SHA)
        print(k,'mass accuracy',float(np.mean((mass[:,1]>=.5)==y[q])),flush=True)


def report(context):
    from exhaustive_family_quorums_n26 import inputs
    from scipy.special import logit
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    raw,y,g,f=data()
    _,cur,_=inputs()
    prior=np.load(OUT/'mitra96.npz')
    np.testing.assert_array_equal(prior['fold'],f)
    np.testing.assert_array_equal(prior['passenger'],raw.PassengerId.to_numpy(str))
    mitra=prior['p']
    features=np.empty((len(y),5))
    for k in range(8):
        with np.load(packet_dir(context)/f'mass{k}.npz') as z:
            q=np.flatnonzero(f==k)
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['labels'],y[z['context']])
            np.testing.assert_array_equal(z['ids'],raw.PassengerId.iloc[q].to_numpy(str))
            features[q]=np.column_stack([z['mean'],z['mass'],z['std']])
    mean=features[:,0]
    mass=features[:,2]
    scores=np.column_stack([mitra,features])
    pred={}
    for k in np.unique(f):
        rows=np.flatnonzero(f==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=192).split(rows,groups=g[rows]):
            t,q=rows[a],rows[b]
            assert not np.isin(g[t],g[q]).any()
            for name,n in [('mean_with_mitra',2),('distribution_with_mitra',6)]:
                x=scores[:,:n].copy()
                x[:,:min(n,5)]=logit(x[:,:min(n,5)].clip(.002,.998))
                scale=StandardScaler().fit(x[t])
                model=LogisticRegression(C=.05,max_iter=450).fit(scale.transform(x[t]).clip(-5,5),y[t])
                pred.setdefault(name,np.empty(len(y),bool))[q]=model.predict(scale.transform(x[q]).clip(-5,5))
    anchor=cur['parent']
    mass_blend=3*logit(mitra.clip(.001,.999))+logit(mass.clip(.001,.999))>=0
    actions=dict(raw_mass=(mass>=.5)!=anchor,new_mass=(mass>=.5)!=anchor,
                 mass_blend_extra=(mass_blend!=anchor)&((mitra>=.5)==anchor),
                 calibrated_extra=(pred['distribution_with_mitra']!=anchor)&(pred['mean_with_mitra']==anchor))
    actions['new_mass'] &= (mean>=.5)==anchor
    def measures(a,m):
        return [int((a&m&(y!=anchor)).sum()),int((a&m&(y==anchor)).sum())]
    r=dict(context=context,checkpoint=SHA,mean_accuracy=float(((mean>=.5)==y).mean()),
           mass_accuracy=float(((mass>=.5)==y).mean()),counts='[fixed,harmed]',
           scopes={s:{k:measures(a,m) for k,a in actions.items()}
                  for s,m in [('all',cur['known']),('common',cur['common']),('new',cur['known']&cur['new'])]})
    table=json.loads(REPORT.read_text()) if REPORT.exists() else {}
    table[str(context)]=r
    REPORT.write_text(json.dumps(table,indent=2)+'\n')
    print(r,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('extract','report'))
    ap.add_argument('--context',type=int,default=1536)
    args=ap.parse_args()
    extract(args.context) if args.action=='extract' else report(args.context)
