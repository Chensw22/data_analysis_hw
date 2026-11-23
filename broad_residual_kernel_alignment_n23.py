"""Weight nonlinear manifest interactions by class evidence replicated in separate training bookings."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS, load
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write

REPORT = ROOT/'reports/nov23_field_robustness'


def blocks(frame):
    bills=frame[BILLS]
    raw=[
        pd.get_dummies(frame[['HomePlanet','Destination','Age','VIP','CryoSleep']],
                       columns=['HomePlanet','Destination','VIP','CryoSleep']).astype(float),
        pd.get_dummies(frame[['Deck','Side','CabinNumber']],columns=['Deck','Side']).astype(float),
        frame[['BookingSize']].astype(float),
        pd.concat([bills.gt(0),bills.isna(),np.log1p(bills.sum(axis=1)).rename('sum')],axis=1).astype(float),
        np.log1p(bills)]
    return [v.fillna(-2).to_numpy(float) for v in raw]


def mappings(parts,t,rng):
    anchors=rng.choice(t,size=40,replace=False)
    distances=[]
    for x in parts:
        scale=x[t].std(0).clip(.3)
        normalized=((x-x[t].mean(0))/scale).clip(-5,5)
        gap=cdist(normalized,normalized[anchors],'sqeuclidean')
        sample=gap[t]
        distances.append(gap/max(.1,np.median(sample[sample>0])))
    coalitions=[c for n in range(1,len(parts)+1) for c in combinations(range(len(parts)),n)]
    kern=[]
    for c in coalitions:
        k=np.exp(-sum(distances[j] for j in c)/len(c))
        ev,vec=np.linalg.eigh((k[anchors]+k[anchors].T)/2)
        b=(k@vec)/np.sqrt(ev.clip(.01))
        b-=b[t].mean(0)
        kern.append((b/max(.2,np.sqrt(np.mean((b[t]**2).sum(1))))).astype(np.float32))
    return coalitions,kern


def classifier(design,y,t,q):
    m=LogisticRegression(C=.35,max_iter=350)
    return m.fit(design[t],y[t]).predict_proba(design[q])[:,1]


def fit(parts,y,groups,t,q):
    coal,phi=mappings(parts,t,np.random.default_rng(452))
    sub=[t[b] for _,b in GroupKFold(3,shuffle=True,random_state=143).split(t,groups=groups[t])]
    for a,b in combinations(sub,2):
        assert not np.intersect1d(groups[a],groups[b]).size
    single=np.column_stack(phi[:len(parts)])
    # Separate the preliminary fit, and the two groups measuring replicated residual evidence.
    # Rotate all three roles before fitting the final classifier on all outer training bookings.
    strength=np.zeros(len(coal))
    for i in range(3):
        a,b,c=sub[i],sub[(i+1)%3],sub[(i+2)%3]
        pred=classifier(single,y,a,np.r_[b,c])
        rb=y[b]-pred[:len(b)]
        rc=y[c]-pred[len(b):]
        for k,ph in enumerate(phi):
            cov1=(ph[b].T@rb)/len(b)
            cov2=(ph[c].T@rc)/len(c)
            strength[k]+=(cov1@cov2)/3
    # Allocate the same total feature weight to every interaction order in all three arms.
    learned=np.ones(len(coal))
    sham=learned.copy()
    rng=np.random.default_rng(773)
    for order in range(1,len(parts)+1):
        ids=np.flatnonzero([len(v)==order for v in coal])
        positive=np.maximum(0,strength[ids])
        learned[ids]=(.25+.75*positive/positive.mean()) if positive.mean()>0 else 1.
        sham[ids]=rng.permutation(learned[ids])
    predictions={'first_order':classifier(single,y,t,q)}
    for name,w in [('uniform',np.ones(len(coal))),('replicated',learned),('permuted',sham)]:
        design=np.column_stack([p*np.sqrt(v) for p,v in zip(phi,w)])
        predictions[name]=classifier(design,y,t,q)
    return predictions,learned


def main():
    row,y,frame,groups,_=load()
    parts=blocks(frame)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        old=z['votes']
    wrong=(old!=y[None,:,None]).all((0,2))
    right=(old==y[None,:,None]).all((0,2))
    out=ART/'nov23_residual_kernel_alignment'
    out.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        scores,weight={},[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(groups,groups[q]))
            assert not np.intersect1d(groups[t],groups[q]).size
            bank,w=fit(parts,y,groups,t,q)
            weight.append(w)
            for k,p in bank.items():
                scores.setdefault(k,np.full(len(y),np.nan))[q]=p
        votes={'reference':ref}
        for k,p in scores.items():
            assert np.isfinite(p).all()
            votes[k]=p>=.5
            votes['quarter_'+k]=np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*p>=.5,ref)
        h={k:v==y for k,v in votes.items()}
        plans[seed]={k:dict(reference=paired(v,h['reference']),uniform=paired(v,h[
            'quarter_uniform' if k.startswith('quarter_') else 'uniform'])) for k,v in h.items()}
        if seed!='v131':
            for k,v in h.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=op[:,0],weights=np.array(weight),**scores)
        print(seed,plans[seed]['quarter_replicated'],flush=True)
    h={k:np.stack(v) for k,v in hist.items()}
    miss=(~h['reference']).sum(0)
    stats={k:dict(common=int((v.all(0)&wrong).sum()),common_right=int((v.all(0)&right).sum()),
                  repeat=int((v.all(0)&(miss==3)).sum()),
                  gain_ci95_pp=interval((v.astype(float)-h['reference']).mean(0),groups)) for k,v in h.items()}
    write(REPORT/'alignment.json',dict(plans=plans,grouped=stats))


if __name__=='__main__':
    with threadpool_limits(2):main()
