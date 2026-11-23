"""Learn from the intact record when training to predict a passenger with unavailable fields."""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_observed_mask_shift_n23 import rows, masked_rows
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write

REPORT=ROOT/'reports/nov23_field_robustness'


def fit(x,y,weight=None):
    mat=xgb.DMatrix(x,label=y,weight=weight,nthread=2)
    params=dict(max_depth=4,eta=.06,reg_lambda=4.,min_child_weight=6,tree_method='hist',nthread=2,
                seed=913,base_score=.5,objective='reg:logistic')
    return xgb.train(params,mat,num_boost_round=275)


def predict_fold(blocks,size,missing,y,groups,t,q,f):
    train=masked_rows(blocks,size,missing,t,t,275+f)
    query=masked_rows(blocks,size,missing,t,q,838+f)
    intact=np.full(len(t),np.nan)
    for a,b in GroupKFold(3,shuffle=True,random_state=309).split(t,groups=groups[t]):
        assert not np.intersect1d(groups[t[a]],groups[t[b]]).size
        teacher=fit(train[0,a],y[t[a]])
        intact[b]=teacher.predict(xgb.DMatrix(train[0,b],nthread=2))
    clean=fit(train[0],y[t]).predict(xgb.DMatrix(query[0],nthread=2))
    assert np.isfinite(intact).all()
    rng=np.random.default_rng(718)
    sham=intact.copy()
    for label in (False,True):
        for incomplete in (False,True):
            ix=np.flatnonzero((y[t]==label)&(missing[t].any(1)==incomplete))
            sham[ix]=rng.permutation(sham[ix])
    w=np.tile(np.array([.5,.125,.125,.125,.125])[:,None],(1,len(t)))
    result={}
    for k,response in [('truth',y[t]),('teacher',.5*y[t]+.5*intact),('shuffled',.5*y[t]+.5*sham)]:
        targets=np.vstack([y[t],np.tile(response,(len(train)-1,1))])
        model=fit(np.concatenate(train),targets.flatten(),w.flatten())
        result[k]=model.predict(xgb.DMatrix(np.concatenate(query),nthread=2)).reshape(len(query),len(q)).T
    return clean,result


def main():
    row,y,frame,g,_=load()
    blocks,missing=rows(frame)
    incomplete=missing.any(1)
    size=frame.BookingSize.to_numpy()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        wrong=(z['votes']!=y[None,:,None]).all((0,2))
        right=(z['votes']==y[None,:,None]).all((0,2))
    scopes=dict(all=np.ones(len(y),bool),complete=~incomplete,incomplete=incomplete,
                joint_wrong=wrong,joint_right=right)
    root=ART/'nov23_privileged_manifest'
    root.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        teacher=np.full(len(y),np.nan)
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            teacher[q],pred=predict_fold(blocks,size,missing,y,g,t,q,int(f))
            for k,p in pred.items():scores.setdefault(k,np.full((len(y),5),np.nan))[q]=p
        votes={'reference':ref}
        diag={}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            full=p[:,0]
            votes[k]=full>=.5
            votes['quarter_'+k]=np.where((op[:,0]>=.5)==ref,.75*op[:,0]+.25*full>=.5,ref)
            votes['missing_'+k]=np.where(incomplete,votes['quarter_'+k],ref)
            synthetic=p[~incomplete,1:]
            gold=y[~incomplete,None]
            diag[k]=dict(masked_teacher_mse=float(np.square(synthetic-teacher[~incomplete,None]).mean()),
                    masked_accuracy=float(((synthetic>=.5)==gold).mean()),
                    masked_logloss=float(-np.log(np.where(gold,synthetic,1-synthetic).clip(1e-8)).mean()))
        h={k:v==y for k,v in votes.items()}
        if seed!='v131':
            for k,v in h.items():hist.setdefault(k,[]).append(v)
        plans[seed]=dict(models={k:{n:paired(v[m],h['reference'][m]) for n,m in scopes.items()}
                                 for k,v in h.items()},synthetic=diag)
        np.savez_compressed(root/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],clean=teacher,**scores)
        print(seed,plans[seed]['models']['missing_teacher']['all'],diag['teacher'],flush=True)
    h={k:np.stack(v) for k,v in hist.items()}
    stats={k:dict(stable={n:int((v.all(0)&m).sum()) for n,m in scopes.items()},
             vs_ref95_pp=interval((v.astype(float)-h['reference']).mean(0),g)) for k,v in h.items()}
    write(REPORT/'privileged.json',dict(plans=plans,grouped=stats))


if __name__=='__main__':
    with threadpool_limits(2):main()
