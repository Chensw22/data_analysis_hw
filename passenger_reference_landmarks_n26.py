"""Learn which labelled reference passengers carry transport information on independent bookings."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data
from full_roster_mistake_influence_n24 import design
from passenger_pair_outcomes_n25 import classifier, features, references
from repeated_archive_misses_n26 import ART, cohort, count

OUT = ART/'nov26_passenger_reference_landmarks'
REPORT = ROOT/'reports/nov23_matched_learning_curve/passenger_reference_landmarks.json'
MODES = ('raw','anonymous','identity','reassigned','mean')


def split_orders(rows, groups, fraction, seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(rows,groups=groups[rows]))
    return rows[a], rows[b]


def select(bank,y,raw_score,rng):
    # Include both expected and surprising reference outcomes at every score level.
    refs=[]
    for label in (False,True):
        subset=bank[y[bank]==label]
        ordered=subset[np.argsort(raw_score[subset],kind='stable')]
        refs.extend(np.concatenate([rng.choice(block,4,replace=False) for block in np.array_split(ordered,8)]))
    return np.array(refs)


def paired_scores(model,z,rows,landmarks,y):
    all_scores=[]
    for start in range(0,len(rows),48):
        q=rows[start:start+48]
        # features() expects both a class and a within-class reference dimension.
        bank=np.broadcast_to(landmarks.reshape(2,-1),(len(q),2,len(landmarks)//2))
        p=model.predict_proba(features(z,q,bank))[:,1].reshape(len(q),-1)
        all_scores.append(np.where(y[landmarks][None,:],p,1-p))
    return np.vstack(all_scores)


def fit(z,y,groups,t,q,seed):
    pool, calibrate=split_orders(t,groups,.25,seed)
    examples, bank=split_orders(pool,groups,.4,seed+1)
    for a,b in ((t,q),(examples,bank),(pool,calibrate)):
        assert not np.isin(groups[a],groups[b]).any()
    z=((z-z[pool].mean(0))/z[pool].std(0).clip(.2)).clip(-6,6).astype('float32')
    direct=classifier().fit(z[pool],y[pool])
    raw=direct.predict_proba(z)[:,1]
    rng=np.random.default_rng(seed+2)
    sampled=rng.choice(examples,min(1500,len(examples)),replace=False)
    donors=references(z,y,bank,sampled,seed+3)
    same=(y[sampled,None,None]==y[donors]).ravel()
    pair=classifier().fit(features(z,sampled,donors),same)
    landmarks=select(bank,y,raw,rng)
    assert len(np.unique(landmarks))==64
    ids=np.r_[calibrate,q]
    scores=paired_scores(pair,z,ids,landmarks,y)
    signed=logit(scores.clip(.005,.995))
    ordinary=logit(raw[ids].clip(.005,.995))[:,None]
    blocks=signed.reshape(len(ids),2,32)
    sorted_blocks=np.sort(blocks,axis=2)
    # Preserve all order statistics within each donor label in both identity controls.
    baseline=np.column_stack([ordinary, sorted_blocks[:,:,::4].reshape(len(ids),-1),
                              sorted_blocks[:,:,-1]])
    residual=(blocks-blocks.mean(2,keepdims=True)).reshape(len(ids),64)
    shuffled=np.take_along_axis(blocks,rng.random(blocks.shape).argsort(2),axis=2)
    placebo=(shuffled-shuffled.mean(2,keepdims=True)).reshape(len(ids),64)
    d=len(calibrate)
    out={}
    for k,x in [('raw',ordinary),('anonymous',baseline),('identity',np.column_stack([baseline,residual])),
                ('reassigned',np.column_stack([baseline,placebo]))]:
        scale=x[:d].std(0).clip(.25)
        x=(x-x[:d].mean(0))/scale
        reg=LogisticRegression(C=.08,max_iter=500,tol=1e-7).fit(x[:d],y[calibrate])
        out[k]=reg.predict_proba(x[d:])[:,1]
    out['mean']=scores[d:].mean(1)
    return out,landmarks,calibrate


def run():
    frame,y,g,fold=data()
    base,blind,known,fresh=cohort(frame,y,fold)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=frame.columns)
    roster=pd.concat([frame,test],ignore_index=True)
    groups=roster.PassengerId.str.split('_').str[0].to_numpy()
    z=design(roster)
    n=len(y)
    np.testing.assert_array_equal(z[:n],design(frame))
    out={k:np.full(len(z),np.nan) for k in MODES}
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(fold!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(fold==k) if k<8 else np.arange(n,len(z))
        path=OUT/f'fold{k}.npz'
        if not path.exists():
            scores,refs,cal=fit(z,y,groups,t,q,8724+k)
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,passenger=roster.PassengerId.iloc[q].to_numpy(str),q=q,t=t,labels=y[t],
                                refs=refs,cal=cal,**scores)
            tmp.replace(path)
        with np.load(path) as saved:
            for key,value in [('passenger',roster.PassengerId.iloc[q]),('q',q),('t',t),('labels',y[t])]:
                np.testing.assert_array_equal(saved[key],value)
            assert not np.isin(groups[saved['refs']],groups[saved['cal']]).any()
            for mode in MODES:out[mode][q]=saved[mode]
        print('fold',k,flush=True)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as saved:
        np.testing.assert_array_equal(saved['passenger'],roster.PassengerId)
        anchor=saved['anchor']
        strength=abs(logit(saved['scores'][:,-3:].mean(1).clip(.01,.99)))
    np.testing.assert_array_equal(anchor[:n],base)
    for mode,p in out.items():assert np.isfinite(p).all()
    signed=np.where(anchor,strength,-strength)
    for control in ('anonymous','reassigned'):
        difference=logit(out['identity'].clip(.001,.999))-logit(out[control].clip(.001,.999))
        out['over_'+control]=expit(signed+difference)
    edits={mode:(p>=.5)!=anchor for mode,p in out.items()}
    edits['identity_only']=edits['identity'] & ~edits['anonymous'] & ~edits['reassigned']
    edits['both_increments']=edits['over_anonymous'] & edits['over_reassigned']
    report=dict(accuracy={k:float(((p[:n]>=.5)==y).mean()) for k,p in out.items() if not k.startswith('over')},rules={})
    for mode,v in edits.items():
        diff=np.where(base!=y,1,-1)*v[:n]
        report['rules'][mode]=dict(test_changes=int(v[n:].sum()),by_fold=[int(diff[(fold==i)&known].sum()) for i in range(8)],
            scopes={k:count(v[:n],m,base!=y) for k,m in [('all',known),('blind',blind),('fresh',known&fresh)]})
    np.savez_compressed(OUT/'aligned.npz',passenger=roster.PassengerId.to_numpy(str),y=y,fold=fold,**out)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['accuracy'],[(k,v['scopes']['all']) for k,v in report['rules'].items()])


if __name__=='__main__':
    with threadpool_limits(2):
        run()
