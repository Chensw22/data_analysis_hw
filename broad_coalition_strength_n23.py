"""Learn the contribution of each set of personal evidence sources at each strength level."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import CATEGORICAL, load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

FIELDS = [
    ['Age', 'CryoSleep'], ['HomePlanet', 'Destination', 'VIP'],
    ['Deck', 'CabinNumber', 'Side'], ['Spa', 'RoomService', 'VRDeck'], ['FoodCourt', 'ShoppingMall'],
]
REPORT = ROOT/'reports/nov23_coalition_strength'


def personal_evidence(frame, y, train, query):
    result = []
    for fields in FIELDS:
        raw = frame[fields].copy()
        for f in fields:
            if f in CATEGORICAL:
                levels = raw[f].iloc[train].unique()
                raw[f] = pd.Categorical(raw[f].where(raw[f].isin(levels)), categories=levels)
        x = pd.get_dummies(raw).to_numpy(float)
        model = HistGradientBoostingClassifier(max_iter=85, max_leaf_nodes=9, min_samples_leaf=24,
                                               learning_rate=.07, l2_regularization=5, early_stopping=False,
                                               random_state=137)
        model.fit(x[train], y[train])
        result.append(logit(model.predict_proba(x[query])[:,1].clip(.006,.994)))
    return np.column_stack(result)


def strength_layers(values):
    # Column S is the width of the threshold range where exactly the sources in S are active.
    ordering = np.argsort(-values, axis=1, kind='stable')
    levels = np.take_along_axis(values, ordering, axis=1)
    widths = levels - np.column_stack([levels[:,1:], np.zeros(len(values))])
    codes = np.bitwise_or.accumulate(1 << ordering, axis=1)
    block = np.zeros((len(values), 2**values.shape[1]-1))
    block[np.arange(len(values))[:,None], codes-1] = widths
    # Integrating the indicator for source j over thresholds recovers its original strength.
    present = ((np.arange(1,block.shape[1]+1)[:,None] >> np.arange(values.shape[1])) & 1)
    np.testing.assert_allclose(block @ present, values, atol=1e-10)
    return block, widths


def readouts(evidence, y, train_size):
    positive, negative = np.maximum(evidence,0), np.maximum(-evidence,0)
    pos, pos_width = strength_layers(positive)
    neg, neg_width = strength_layers(negative)
    raw = np.c_[positive, negative]
    shape = np.c_[raw,pos_width,neg_width]
    # The sham keeps the strength distribution, signs, and additive evidence columns.
    # Only the field names of the additional set-valued columns are permuted per person.
    rng = np.random.default_rng(477)
    perm = rng.random(evidence.shape).argsort(axis=1)
    sham = [strength_layers(np.take_along_axis(v,perm,axis=1))[0] for v in (positive,negative)]
    designs = dict(additive=raw, anonymous=shape, named=np.c_[shape,pos,neg], shuffled=np.c_[shape,*sham])
    predictions = {}
    for k,x in designs.items():
        model = LogisticRegression(C=.12, max_iter=500, tol=1e-6).fit(x[:train_size], y)
        predictions[k] = model.predict_proba(x[train_size:])[:,1]
    return predictions


def fold_scores(frame,y,g,t,q):
    assert not np.intersect1d(g[t],g[q]).size
    estimates = {}
    for a,b in GroupKFold(2,shuffle=True,random_state=549).split(t,groups=g[t]):
        source, calibration = t[a],t[b]
        assert not np.intersect1d(g[source],g[calibration]).size
        evidence = personal_evidence(frame,y,source,np.r_[calibration,q])
        for key,p in readouts(evidence,y[calibration],len(calibration)).items():
            estimates.setdefault(key,[]).append(p)
    return {k:np.mean(v,axis=0) for k,v in estimates.items()}


def main():
    row,y,frame,g,_ = load()
    wrong,right,hard = cohorts(row,y)
    solo = frame.BookingSize.to_numpy()==1
    out = ART/'nov23_coalition_strength'
    out.mkdir(exist_ok=True)
    plans,history = {},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_ = read_opinions(seed,row,y)
        predictions = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f) & ~np.isin(g,g[q]))
            for k,p in fold_scores(frame,y,g,t,q).items():
                predictions.setdefault(k,np.full(len(y),np.nan))[q] = p
        hits = {'reference':ref==y}
        for k,p in predictions.items():
            assert np.isfinite(p).all()
            hits[k] = (p>=.5)==y
            h = (.75*op[:,0]+.25*p >= .5)
            hits['quarter_'+k] = np.where(ref==(op[:,0]>=.5),h,ref)==y
        change = logit(op[:,0].clip(.002,.998))+logit(predictions['named'])-logit(predictions['anonymous'])
        hits['named_increment'] = np.where(ref==(op[:,0]>=.5),change>=0,ref)==y
        metrics = {k:{name:paired(h[mask],hits['reference'][mask]) for name,mask in
                      [('all',np.ones(len(y),bool)),('solo',solo),('hard',hard),('common_right',right)]}
                   for k,h in hits.items()}
        plans[seed] = metrics
        if seed!='v131':
            for k,v in hits.items():history.setdefault(k,[]).append(v)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=op[:,0],ref=ref,**predictions)
        print(seed,{k:d['all']['fixed']-d['all']['harmed'] for k,d in metrics.items()},flush=True)
    baseline = np.stack(history['reference'])
    stable = {k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),
                     persistent_hard=int((np.all(v,axis=0)&hard).sum()),
                     solo_persistent_hard=int((np.all(v,axis=0)&hard&solo).sum()),
                     hard_ever=int((np.any(v,axis=0)&hard).sum()),
                     common_right_harmed=int((~np.any(v,axis=0)&right).sum()),
                     gain95_pp=interval((np.stack(v).astype(float)-baseline).mean(0),g)) for k,v in history.items()}
    write(REPORT/'results.json',dict(plans=plans,stable=stable,n_solo=int(solo.sum()),n_hard_solo=int((hard&solo).sum())))


if __name__=='__main__':
    with threadpool_limits(1):main()
