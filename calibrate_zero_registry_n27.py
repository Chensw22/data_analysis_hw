"""Test domain-specific registry evidence across the full no-positive-bill population."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from extend_six_payment_states_n27 import OUT,ART,ROOT,FEES
from full_roster_booking_validation_n23 import data
from repeated_archive_misses_n26 import cohort,count


def fit(x,offset,y,t,q):
    mid=x[t].mean(0)
    scale=x[t].std(0).clip(.08)
    a=np.column_stack([np.ones(len(t)),(x[t]-mid)/scale])
    def loss(w):
        z=offset[t]+a@w
        return (np.logaddexp(0,z).sum()-y[t]@z+10*(w@w),a.T@(expit(z)-y[t])+20*w)
    opt=minimize(loss,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B')
    assert opt.success,opt.message
    return offset[q]+opt.x[0]+((x[q]-mid)/scale)@opt.x[1:]


def run():
    raw,y,g,f=data()
    n=len(y)
    _,blind,known,fresh=cohort(raw,y,f)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    ids=frame.PassengerId.to_numpy(str)
    groups=frame.PassengerId.str[:4].to_numpy()
    zero=~frame[FEES].gt(0).any(axis=1).to_numpy()
    sizes=frame.groupby(groups).PassengerId.transform('size').to_numpy()
    with np.load(OUT/'actions.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        six,family,base=z['six'],z['family'],z['base']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['names'][-3:],['Mitra','TabICL fare','Nori fare'])
        native=logit(z['scores'][:,-3:].mean(1).clip(.005,.995))
    views={}
    for key in ('specialist','matched_prior'):
        rep=np.full((len(ids),2),np.nan)
        for k in range(9):
            with np.load(OUT/f'fold{k}.npz') as z:
                np.testing.assert_array_equal(z['ids'] if 'ids' in z else z['passenger'],ids[z['q']])
                rep[z['q']]=z[key]
        assert np.isfinite(rep).all()
        views[key]=rep
    odds=lambda p:logit(np.clip(p,.005,.995))
    common=np.column_stack([family.mean(1),odds(six[:,0]),odds(six[:,3:]),frame.CryoSleep.eq(True),
         frame.CryoSleep.isna(),frame[FEES].isna().sum(axis=1),np.log1p(sizes)])
    design={key:np.column_stack([common,odds(rep)]) for key,rep in dict(frozen=six[:,1:3],**views).items()}
    design['common']=common
    inner=partition(g,'zero_registry_weights',4)
    scores={}
    for key,x in design.items():
        s=np.full(len(ids),np.nan)
        test=[]
        for k in range(8):
            for j in [0,1,2,3,None]:
                t=np.flatnonzero((f==k)&zero[:n]&((inner!=j) if j is not None else True))
                q=np.flatnonzero((f==k)&(inner==j)&zero[:n]) if j is not None else np.flatnonzero(zero[n:])+n
                assert not np.isin(groups[t],groups[q]).any()
                p=fit(x,native,y,t,q)
                if j is None:test.append(p)
                else:s[q]=p
        s[np.flatnonzero(zero[n:])+n]=np.mean(test,axis=0)
        assert np.isfinite(s[zero]).all()
        scores[key]=s
    anchor=np.where(base,1.,-1.)*abs(native)
    flags={k:zero & ((p>=0)!=base) for k,p in scores.items()}
    for key in ('frozen','common','matched_prior'):
        flags['relative_'+key]=zero & ((anchor+scores['specialist']-scores[key]>=0)!=base)
    flags['all_relative']=flags['relative_frozen']&flags['relative_common']&flags['relative_matched_prior']&flags['specialist']
    flags['exclusive']=flags['specialist']&~flags['frozen']&~flags['matched_prior']
    scopes=dict(all=zero[:n]&known,fresh=zero[:n]&known&fresh,blind=blind&zero[:n],
                single=zero[:n]&known&(sizes[:n]==1),pair=zero[:n]&known&(sizes[:n]==2),larger=zero[:n]&known&(sizes[:n]>=3))
    result={}
    for k,act in flags.items():
        delta=act[:n]*np.where(base[:n]!=y,1,-1)
        result[k]=dict(test_changes=int(act[n:].sum()),by_fold=[int(delta[known&(f==j)].sum()) for j in range(8)],
                       scopes={name:count(act[:n],m,base[:n]!=y) for name,m in scopes.items()})
    np.savez_compressed(OUT/'calibrated.npz',passenger=ids,y=y,fold=f,base=base,**flags)
    (ROOT/'reports/nov23_matched_learning_curve/calibrate_zero_registry.json').write_text(json.dumps(result,indent=2)+'\n')
    print([(k,v['scopes']['all'],v['scopes']['fresh']) for k,v in result.items()])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
