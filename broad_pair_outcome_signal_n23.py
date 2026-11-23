"""Learn departures from independent outcome agreement on pairs of different bookings."""
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_refit_dissent_n23 import features
from broad_training_membership_n23 import model

REPORT = ROOT/'reports/nov23_release_vote_information'


def construct_pairs(x, margin, i, j):
    a, b = margin[i], margin[j]
    # Symmetric across passenger order; keep age/service ranges and categorical agreements.
    return np.column_stack([np.minimum(a, b), np.maximum(a, b), a*b,
                            np.minimum(x[i], x[j]), np.maximum(x[i], x[j])])


def compare_pairs(x, y, groups, t, q, seed):
    pairs_train, hold = next(GroupKFold(3, shuffle=True, random_state=196).split(t,groups=groups[t]))
    a, hold = t[pairs_train], t[hold]
    assert not np.intersect1d(groups[a],groups[hold]).size
    margin = np.full(len(y),np.nan)
    for tr, va in GroupKFold(3,shuffle=True,random_state=513).split(a,groups=groups[a]):
        m = model(180).fit(x[a[tr]],y[a[tr]])
        margin[a[va]] = 2*m.predict_proba(x[a[va]])[:,1]-1
    point = model(180).fit(x[a],y[a])
    for ix in [hold,q]:
        margin[ix]=2*point.predict_proba(x[ix])[:,1]-1
    assert not np.intersect1d(groups[t],groups[q]).size
    sign=2*y.astype(float)-1
    rand=np.random.default_rng(seed)
    i = np.repeat(a,14)
    j = rand.choice(a,len(i))
    distinct = groups[i]!=groups[j]
    i,j = i[distinct],j[distinct]
    # Sham outcomes keep the class frequency inside narrow out-of-fold risk bins.
    sham=sign.copy()
    for cell in np.array_split(a[np.argsort(margin[a])],12):
        sham[cell]=rand.permutation(sham[cell])
    raw=construct_pairs(x,margin,i,j)
    # Stratified reference selection spans the risk range; their outcomes were unseen by both learners.
    anchors=np.concatenate([rand.choice(c,min(12,len(c)),replace=False)
                            for c in np.array_split(hold[np.argsort(margin[hold])],10)])
    sampled=sign[anchors]
    expected=margin[anchors]
    normalizer=sampled@expected
    assert normalizer>0
    result={'point':(1+margin[q])/2}
    for key, response in [('same_outcome',sign),('shuffled',sham)]:
        target=response[i]*response[j]-margin[i]*margin[j]
        estimator=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=50,
                            learning_rate=.07,l2_regularization=8.,early_stopping=False,random_state=548)
        estimator.fit(raw,target)
        residual=[]
        for batch in np.array_split(q,int(np.ceil(len(q)/125))):
            r=construct_pairs(x,margin,np.repeat(batch,len(anchors)),np.tile(anchors,len(batch)))
            corr=estimator.predict(r).reshape(len(batch),len(anchors))
            residual.extend(corr@sampled/normalizer)
        result[key]=((1+margin[q]+np.array(residual))/2).clip(.002,.998)
    return result


def main():
    row,y,frame,g,_=load()
    x=features(frame)[0]
    output=ART/'nov23_pair_outcome_signal'
    output.mkdir(exist_ok=True)
    plans, hist, common = {},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        fold, ref, bank, _=read_opinions(seed,row,y)
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for k,p in compare_pairs(x,y,g,t,q,516+int(f)).items():
                scores.setdefault(k,np.full(len(y),np.nan))[q]=p
        assert all(np.isfinite(p).all() for p in scores.values())
        methods={'reference':ref}
        for k,p in scores.items():
            methods[k]=p>=.5
            # Compare confidence changes at the same baseline probability.
            delta=logit(p.clip(.002,.998))-logit(scores['point'].clip(.002,.998))
            adjusted=logit(bank[:,0].clip(.002,.998))+delta
            methods['transfer_'+k]=np.where(ref==(bank[:,0]>=.5), adjusted>=0,ref)
        h={k:v==y for k,v in methods.items()}
        if seed!='v131':
            common.append(((bank>=.5)!=y[:,None]).all(1))
            for k,v in h.items():
                hist.setdefault(k,[]).append(v)
        plans[seed]={k:dict(vs_reference=paired(v,ref==y),vs_point=paired(v,h['point']))
                     for k,v in h.items()}
        np.savez_compressed(output/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,**scores)
        print(seed,plans[seed]['transfer_same_outcome']['vs_reference'],flush=True)
    common=np.stack(common).all(0)
    assert common.sum()==332
    hits={k:np.stack(v) for k,v in hist.items()}
    reference=hits['reference']
    miss=(~reference).sum(0)
    stable={}
    for k,v in hits.items():
        stable[k]=dict(common=int((v.all(0)&common).sum()),
                       by_baseline_error_count=[int((v.all(0)&(miss==j)).sum()) for j in range(4)],
                       net95_pp=interval((v.astype(float)-reference).mean(0),g))
    write(REPORT/'pair_outcomes.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):main()
