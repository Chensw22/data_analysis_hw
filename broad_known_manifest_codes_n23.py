"""Predict outcome codes jointly and decode them using actually observed passenger attributes."""
import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from broad_fixed_vote_totals_n23 import REPORT
from broad_opposing_passengers_n23 import evaluate
from exaone_validation_data import CATEGORICAL

CHARACTERS=np.array([[(-1)**((state&mask).bit_count()) for mask in range(16)] for state in range(16)])


def simplex(raw):
    # Euclidean projection onto nonnegative probabilities summing to one per row.
    ordered=np.sort(raw,axis=1)[:,::-1]
    shifts=(ordered.cumsum(1)-1)/np.arange(1,raw.shape[1]+1)
    active=(ordered>shifts).sum(1)-1
    return np.maximum(raw-shifts[np.arange(len(raw)),active,None],0.)


def decode(joint,state,fallback):
    rows=np.arange(len(joint))
    negative=joint[rows,2*state]
    positive=joint[rows,2*state+1]
    mass=positive+negative
    return np.divide(positive,mass,out=fallback.copy(),where=mass>1e-9)


def train(frame,target,t,q,loss):
    model=CatBoostClassifier(iterations=240,depth=5,learning_rate=.065,l2_leaf_reg=4,
                             loss_function=loss,random_seed=485,thread_count=2,
                             verbose=False,allow_writing_files=False)
    model.fit(frame.iloc[t],target,cat_features=CATEGORICAL)
    return model.predict_proba(frame.iloc[q]),model.classes_


def run_fold(frame,groups,y,t,q):
    asleep=frame.CryoSleep.eq('True').to_numpy(int)
    adverse=frame[['Spa','VRDeck','RoomService']].gt(0).any(axis=1).to_numpy(int)
    multi=frame.BookingSize.gt(1).to_numpy(int)
    state=asleep+2*adverse+4*multi
    combined=2*state[t]+y[t].astype(int)
    bits=((1-CHARACTERS[combined,1:])//2).astype(int)
    # A few physically inconsistent states can be absent. Each individual bit has
    # both values in training, even when the full sixteen-state table is sparse.
    assert (bits.max(0)==1).all() and (bits.min(0)==0).all()
    soft,_=train(frame,bits,t,q,'MultiLogloss')
    assert soft.shape==(len(q),15)
    moments=np.column_stack([np.ones(len(q)),1-2*soft])
    raw=moments@CHARACTERS.T/16
    np.testing.assert_allclose(raw.sum(1),1.,atol=1e-10)
    joint=simplex(raw)
    np.testing.assert_allclose(joint.sum(1),1.,atol=1e-10)
    # Each odd mask contains Y. Remove the known contribution of the other bits.
    flipped=CHARACTERS[2*state[q],1::2]<0
    decoded=np.where(flipped,1-soft[:,::2],soft[:,::2])
    basic,_=train(frame,y[t].astype(int),t,q,'Logloss')
    categorical,seen=train(frame,combined,t,q,'MultiClass')
    direct_joint=np.zeros_like(joint)
    direct_joint[:,seen.astype(int)]=categorical
    marginal=joint[:,1::2].sum(1).clip(0.,1.)
    ordinary=direct_joint[:,1::2].sum(1).clip(0.,1.)
    scores=dict(binary=basic[:,1],shared_binary=soft[:,0],code_mean=decoded.mean(1),
                code_conditional=decode(joint,state[q],decoded.mean(1)),code_marginal=marginal,
                ordinary_conditional=decode(direct_joint,state[q],ordinary),ordinary_marginal=ordinary)
    return scores,dict(passengers=len(q),mean_negative_mass=float(np.maximum(-raw,0.).sum(1).mean()))


if __name__=='__main__':
    with threadpool_limits(2):
        evaluate('known_manifest_codes',run_fold,REPORT)
