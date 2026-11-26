"""Compare continuous expert confidence with hard consensus for separating correct and mistaken passengers."""
import json
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from kaggle_workflow import sha256
from repeated_archive_misses_n26 import ART, OUT as BANK, cohort, count

OUT = ART/'nov26_continuous_archive_margins'
REPORT = ROOT/'reports/nov23_matched_learning_curve/continuous_archive_margins.json'


def representation(p, mode):
    odds = logit(p.clip(.01,.99))
    if mode == 'votes':
        return np.sign(odds)
    if mode == 'distribution':
        return np.quantile(odds,[0,.1,.25,.5,.75,.9,1],axis=1).T
    center = np.median(odds,axis=1,keepdims=True)
    return np.column_stack([center, odds-center])


def whiten(x, t, q):
    center = x[t].mean(0)
    scale = np.maximum(x[t].std(0), .15)
    a = (x[t]-center)/scale
    values,vectors = np.linalg.eigh(a.T@a/len(t))
    basis = vectors[:,-6:] / np.sqrt(.5+values[-6:])
    return a@basis, ((x[q]-center)/scale)@basis


def fit(x, features, y, t, q):
    a,b = whiten(x,t,q)
    # Consensus direction and the presence of an opposing configuration are observable on every query.
    side = features.mean(1)>=.5
    contrast = np.where(side[:,None], .5-features, features-.5).max(1)
    band = 2*side+(contrast>0)
    out = np.empty(len(q))
    for region in np.unique(band[q]):
        it = band[t]==region
        iq = band[q]==region
        vote = np.where(y[t],1.,-1.)
        log_ratio = np.zeros(iq.sum())
        counts = []
        for target in (False,True):
            global_class = a[y[t]==target]
            cells = a[it & (y[t]==target)]
            # Ten same-class observations from the wider calibration sample regularize rare cells.
            prior_n = 10
            mean = (cells.sum(0)+prior_n*global_class.mean(0))/(len(cells)+prior_n)
            within = cells-mean
            covariance = (within.T@within+prior_n*np.cov(global_class.T))/(len(cells)+prior_n)
            covariance = .75*covariance+.25*np.eye(len(mean))
            delta = b[iq]-mean
            qf = (np.linalg.solve(covariance, delta.T).T*delta).sum(1)
            sign,ld=np.linalg.slogdet(covariance)
            assert sign > 0
            log_ratio += (.5 if not target else -.5)*(ld+qf)
            counts.append(len(cells))
        prior = (counts[1]+2*(1+np.sum(vote>0))/(len(t)+2))/(sum(counts)+2)
        out[iq] = expit(logit(prior)+log_ratio)
    return out


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    with np.load(BANK/'aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_array_equal(z['y'],y)
        wide=z['full']
    _,_,_,_,names,p,*_=panels_and_reference()
    freeze=ART/'proposal_n26_dispute_band/frozen_votes.npz'
    meta=json.loads((ROOT/'reports/proposal_n26_dispute_band/manifest.json').read_text())
    assert sha256(freeze)==meta['model_sha256']
    with np.load(freeze) as z:
        np.testing.assert_array_equal(z['names'],names)
        test=z['p']
        test_ids=z['passenger']
    n=len(y)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        assert len(z['anchor'])==n+len(test)
        np.testing.assert_array_equal(z['passenger'][n:],test_ids)
        anchors=z['anchor']
        native=z['scores'][:,-3:].mean(1)
    np.testing.assert_array_equal(anchors[:n],base)
    inner=partition(g,'continuous_contrast_validation',4)
    no_challenge=blind & ~(((wide-.5)*np.where(base,-1.,1.)[:,None])>0).any(1)
    scopes=dict(all=known,fresh=known & fresh,blind=blind,never_challenged=no_challenge)
    report={}
    cache={}
    for bank,scores in [('wide',wide),('deployable',np.vstack([p,test]))]:
        size=len(scores)
        opinions={}
        out={}
        for mode in ('votes','distribution','continuous'):
            x=representation(scores,mode)
            values=np.full(size,np.nan)
            test_prob=[]
            for k in np.unique(f):
                for s in [0,1,2,3,None]:
                    t=np.flatnonzero((f==k)&((inner!=s) if s is not None else True))
                    q=np.flatnonzero((f==k)&(inner==s)) if s is not None else np.arange(n,size)
                    if not len(q):continue
                    if s is not None: assert not np.isin(g[t],g[q]).any()
                    prob=fit(x,scores,y,t,q)
                    if s is None:test_prob.append(prob)
                    else:values[q]=prob
            if size>n:values[n:]=np.mean(test_prob,axis=0)
            assert np.isfinite(values).all()
            opinions[mode]=values
        incumbent=anchors[:size]
        offset=np.where(incumbent,1.,-1.)*abs(logit(native[:size].clip(.01,.99)))
        for baseline in ('votes','distribution'):
            gain=logit(opinions['continuous'].clip(.001,.999))-logit(opinions[baseline].clip(.001,.999))
            opinions['increment_over_'+baseline]=expit(offset+gain)
        for mode,prob in opinions.items():
            edit=(prob>=.5)!=incumbent
            delta=edit[:n]*np.where(base!=y,1,-1)
            error_score=np.where(base,1-prob[:n],prob[:n])
            out[mode]=dict(test_edits=int(edit[n:].sum()) if size>n else None,
                scopes={k:{**count(edit[:n],m,base!=y), 'average_precision':float(average_precision_score(base[m]!=y[m],error_score[m])),
                            'prevalence':float((base[m]!=y[m]).mean())} for k,m in scopes.items()},
                by_fold=[int(delta[known&(f==k)].sum()) for k in np.unique(f)])
            cache[bank+'_'+mode]=prob
        report[bank]=out
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'scores.npz',passenger=np.r_[raw.PassengerId.to_numpy(str),test_ids],fold=f,y=y,**cache)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print([(k,v['continuous']['scopes']['blind'],v['increment_over_votes']['scopes']['all']) for k,v in report.items()])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
