"""Compare a bill basket with the joint class distributions of other bookings using the energy score."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from full_roster_booking_validation_n23 import ROOT, data

BILLS = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
REPORT = ROOT/'reports/nov23_matched_learning_curve/conditional_basket_energy.json'
FOLDER = ROOT/'artifacts/nov26_conditional_basket_energy'


def projections(bills):
    # Log dollars distinguish a single large charge; pairwise log contrasts test the allocation across services.
    v=np.log1p(np.nan_to_num(bills))/8
    pairs=list(combinations(range(5),2))
    diff=np.column_stack([(v[:,a]-v[:,b])/np.sqrt(2) for a,b in pairs])
    sums=np.column_stack([(v[:,a]+v[:,b])/np.sqrt(2) for a,b in pairs])
    return np.column_stack([v, diff, sums])


def crps(bank, query):
    # CRPS(X, q) = E|X-q| - .5 E|X-X'|. Exact empirical sums include diagonal pairs.
    n,p=bank.shape
    s=np.sort(bank,axis=0)
    cum=np.vstack([np.zeros(p),np.cumsum(s,axis=0)])
    scatter=((2*np.arange(n)+1-n)@s)/(n*n)
    split=np.column_stack([np.searchsorted(s[:,j],query[:,j]) for j in range(p)])
    lower=cum[split,np.arange(p)]
    energy=(query*(2*split-n)+cum[-1]-2*lower)/n-scatter
    return energy


def class_contrast(bills, y, t, q, rng):
    query=projections(bills[q])
    terms=[]
    for side in (False, True):
        b=bills[t[y[t]==side]]
        shuffled=np.column_stack([rng.permutation(col) for col in b.T])
        actual=crps(projections(b),query)
        control=crps(projections(shuffled),query)
        np.testing.assert_allclose(actual[:,:5],control[:,:5],rtol=0,atol=1e-12)
        # Positive differences favor a transported passenger.
        terms.append(np.column_stack([actual[:,:5],actual[:,5:15].mean(1),actual[:,15:].mean(1),
                                      control[:,5:15].mean(1),control[:,15:].mean(1)]))
    return terms[0]-terms[1]


def make_evidence(raw,y,g,t,q,seed):
    bills=raw[BILLS].to_numpy(float)
    total=raw[BILLS].sum(axis=1,min_count=1)
    frame=pd.DataFrame(dict(sleep=raw.CryoSleep,planet=raw.HomePlanet,dest=raw.Destination,
                            deck=raw.Cabin.str.split('/').str[0],solo=~pd.Series(g).duplicated(keep=False),
                            total=pd.cut(total,[-1,0,250,750,1500,3000,np.inf])))
    frame=frame.astype('string').fillna('?')
    rng=np.random.default_rng(seed)
    base=class_contrast(bills,y,t,q,rng)
    scopes=[['sleep'],['sleep','planet','total'],['sleep','planet','dest','deck','solo','total']]
    result=[base]
    for cols in scopes:
        code=frame.groupby(cols,sort=False).ngroup().to_numpy()
        local=result[-1].copy()
        for key in np.unique(code[q]):
            train=t[code[t]==key]
            n=np.bincount(y[train].astype(int),minlength=2).min()
            if n<3: continue
            mask=code[q]==key
            local[mask]+=(n/(n+24))*(class_contrast(bills,y,train,q[mask],rng)-local[mask])
        result.append(local)
    return np.stack(result,axis=1)


def count(a,m,f):
    return dict(rows=int(m.sum()),fixed=int((a[m]>0).sum()),harmed=int((a[m]<0).sum()),
                net_by_fold=[int(a[m&(f==k)].sum()) for k in range(8)])


def run():
    raw,y,g,f=data()
    s=np.array([[0.],[.2],[.5],[.8]])
    np.testing.assert_allclose(crps(s,s),np.abs(s-s.T).mean(0)[:,None]-.5*np.abs(s-s.T).mean())
    audit=pd.read_csv(ROOT/'artifacts/nov25_within_bin_bill_precision/passenger_error_audit.csv',dtype={'PassengerId':str})
    np.testing.assert_array_equal(audit.PassengerId,raw.PassengerId)
    np.testing.assert_array_equal(audit.booking_fold,f)
    np.testing.assert_array_equal(audit.transported,y)
    vote=audit.original_votes_true.to_numpy()
    minor=np.minimum(vote,68-vote)
    new=audit.new_order.to_numpy()
    hard=(vote==0)&~pd.Series(g).duplicated(keep=False).to_numpy()&raw.CryoSleep.eq(False).to_numpy()
    groups={'all':np.ones(len(y),bool),'unanimous':minor==0,'awake_solo_false':hard,'disputed':minor>=4}
    probabilities={}
    FOLDER.mkdir(exist_ok=True)
    for k in range(8):
        t,q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        assert not np.isin(g[t],g[q]).any()
        v=make_evidence(raw,y,g,t,q,839+k)
        assert np.isfinite(v).all()
        with np.load(ROOT/f'artifacts/nov25_grouped_peer_teacher/release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            base=z['native']
        anchor=logit(np.clip(base,.002,.998))[:,None]
        # Constant scaling makes ridge strength comparable between quantile-range scores and logits.
        def design(cols):return np.column_stack([anchor,8*v[:,:,cols].reshape(len(q),-1)])
        models={'anchor':anchor,'marginals':design(np.arange(5)),'shuffled':design([0,1,2,3,4,7,8]),
                'joint':design(range(7)), 'dependence_only':np.column_stack([anchor,8*(v[:,:,5:7]-v[:,:,7:9]).reshape(len(q),-1)])}
        models['basket_alone']=8*v[:,:,:7].reshape(len(q),-1)
        models['shuffled_alone']=8*v[:,:,[0,1,2,3,4,7,8]].reshape(len(q),-1)
        protocols={'full':list(GroupKFold(4,shuffle=True,random_state=3207).split(q,groups=g[q])),
                   'new':[(np.flatnonzero(~new[q]),np.flatnonzero(new[q]))]}
        for protocol,splits in protocols.items():
            for a,b in splits:
                assert not np.isin(g[q[a]],g[q[b]]).any()
                for name,x in models.items():
                    model=LogisticRegression(C=.2,max_iter=400).fit(x[a],y[q[a]])
                    p=model.predict_proba(x[b])[:,1]
                    probabilities.setdefault((protocol,name),[]).append((q[b],p))
        print(k,flush=True)
    known=audit.reference_nine.notna().to_numpy()
    nine=audit.reference_nine.astype('boolean').fillna(False).to_numpy(bool)
    report={}
    for (protocol,name),blocks in probabilities.items():
        q,p=(np.concatenate([v[i] for v in blocks]) for i in range(2))
        np.testing.assert_array_equal(np.sort(q),np.flatnonzero(new) if protocol=='new' else np.arange(len(y)))
        a=((p>=.5)==y[q]).astype(int)-(nine[q]==y[q])
        scores={}
        for n,m in groups.items():
            keep=known[q]&m[q]
            scores[n]=count(a,keep,f[q])
            scores[n]['auc']=float(roc_auc_score(y[q][keep],p[keep]))
        report.setdefault(protocol,{})[name]=scores
        np.savez_compressed(FOLDER/f'{protocol}_{name}.npz',q=q,p=p)
        print(protocol,name,[(n,scores[n]['fixed'],scores[n]['harmed']) for n in groups])
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
