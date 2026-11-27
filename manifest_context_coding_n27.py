"""Predict transport by the class-specific code length of a complete passenger manifest."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaln,expit,logit,logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from export_directional_foundation_n27 import NAME
from six_way_against_scored_nine_n26 import test_decisions
from directional_foundation_transfer_n27 import calibrate
from booking_mix_confidence_transfer_n26 import partition

OUT=ART/'nov27_manifest_context_coding'
REPORT=ROOT/'reports/nov23_matched_learning_curve/manifest_context_coding_n27.json'
CATS=['HomePlanet','Destination','VIP','CryoSleep','Deck','Side']
FEES=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
ORDERS=[['HomePlanet','Deck','Side','CabinNumber','group_size','Destination','Age','VIP','CryoSleep',*FEES],
        ['CryoSleep','Age','VIP','group_size',*FEES[::-1],'HomePlanet','Destination','Deck','Side','CabinNumber']]


def frame_fields(raw,g):
    frame=raw.drop(columns=['PassengerId','Cabin']).copy()
    frame[['Deck','CabinNumber','Side']]=raw.Cabin.str.split('/',expand=True)
    frame.CabinNumber=pd.to_numeric(frame.CabinNumber)
    frame['group_size']=pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
    assert set(frame)==set(ORDERS[0])==set(ORDERS[1])
    return frame


def coding(frame,t):
    bits={}; coarse={}; matrix=[]
    for k in ORDERS[0]:
        s=frame[k]
        if k in CATS:
            levels=sorted(s.iloc[t].dropna().astype(str).unique())
            value=pd.Categorical(s.astype(str),categories=levels).codes.astype(int)+2
            value[s.isna()]=0   # 1 is reserved for unseen symbols.
            width=max(1,int(value.max()).bit_length())
            seq={v:format(int(v),'0'+str(width)+'b') for v in np.unique(value)}
            symbol=value
        else:
            value=s.fillna(-1).astype(int).to_numpy()
            np.testing.assert_array_equal(value[s.notna()],s[s.notna()].astype(float))
            def number(v):
                if v<0:return '0'
                if v==0:return '10'
                binary=format(v,'b')
                # Elias gamma encodes positive integers exactly after the two observation gates.
                return '11'+'0'*(len(binary)-1)+binary
            seq={v:number(v) for v in np.unique(value)}
            if k in FEES:edges=[0,1,150,650,1000,2000]
            elif k=='Age':edges=[0,13,21,35,55]
            elif k=='CabinNumber':edges=[0,175,450,750,1200]
            else:edges=[0,2,3,5]
            symbol=np.searchsorted(edges,value,side='right')
        matrix.append(value);coarse[k]=symbol
        # Encode the complete word in an integer with a leading sentinel one.
        ordered, ix=np.unique(value,return_inverse=True)
        words=np.array([int('1'+seq[v],2) for v in ordered],np.int64)[ix]
        lengths=np.array([len(seq[v]) for v in ordered],np.int64)[ix]
        bits[k]=(words,lengths)
    return bits,coarse,np.column_stack(matrix)


def step(bit,prefix,ctx,labels,train):
    # Each successive partition refines a context with an earlier observable field.
    # KT evidence determines whether to pool the children of that node or to split.
    partitions=[]
    _,nodes=np.unique(prefix,return_inverse=True)
    links=[]
    for feature in ctx:
        partitions.append(nodes)
        key=nodes*(1+feature.max())+feature
        _,first,nodes=np.unique(key,return_index=True,return_inverse=True)
        links.append(partitions[-1][first])
    partitions.append(nodes)
    answers=[]
    for label in (0,1):
        use=train&(labels==label)
        deeper=None
        for depth in reversed(range(len(partitions))):
            node=partitions[depth]
            size=int(node.max())+1
            total=np.bincount(node[use],minlength=size)
            ones=np.bincount(node[use],weights=bit[use],minlength=size)
            pt=(ones+.5)/(total+1.)
            marginal=betaln(ones+.5,total-ones+.5)-betaln(.5,.5)
            if deeper is None:
                evidence=marginal
                prediction=pt[node]
            else:
                children=np.bincount(links[depth],weights=deeper,minlength=size)
                logits=marginal-children
                stop=expit(logits[node])
                prediction=stop*pt[node]+(1-stop)*prediction
                evidence=np.logaddexp(marginal,children)-np.log(2.)
            deeper=evidence
        observed=lambda prob:np.where(bit,prob,1-prob)
        answers.append(np.column_stack([np.log(observed(pt[node])),np.log(observed(prediction))]))
    return np.stack(answers,axis=1)


def scores(bits,coarse,t,q,y):
    ix=np.r_[t,q];tr=np.arange(len(ix))<len(t)
    target=np.zeros(len(ix),int);target[tr]=y[t]
    orders=[]
    for order in ORDERS:
        ll=np.zeros((len(ix),2,2))
        prev=[]
        for k in order:
            words,lengths=(v[ix] for v in bits[k])
            # Prefer immediate predecessors, then add early registration fields as longer contexts.
            refs=list(dict.fromkeys(prev[::-1][:2]+[h for h in ['CryoSleep','HomePlanet','Deck','Age'] if h in prev]))[:4]
            for j in range(int(lengths.max())):
                rows=np.flatnonzero(lengths>j)
                prefix=words[rows]>>(lengths[rows]-j)
                bit=(words[rows]>>(lengths[rows]-1-j))&1
                ll[rows]+=step(bit,prefix,[coarse[h][ix[rows]] for h in refs],target[rows],tr[rows])
            prev.append(k)
        orders.append(ll[~tr])
    # A mixture over the two possible record serialization orders defines a normalized likelihood.
    np.testing.assert_allclose(orders[0][:,:,0],orders[1][:,:,0],rtol=0,atol=1e-9)
    mixed=logsumexp(np.stack(orders),axis=0)-np.log(len(orders))
    prior=logit((y[t].sum()+.5)/(len(t)+1.))
    return expit(prior+mixed[:,1,:]-mixed[:,0,:])


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    frame=pd.concat([raw,test],ignore_index=True)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    fields=frame_fields(frame,groups)
    ids=frame.PassengerId.to_numpy(str)
    n=len(y)
    probs=np.empty((len(frame),3))
    OUT.mkdir(exist_ok=True)
    for k in range(9):
        t=np.flatnonzero(f!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(f==k) if k<8 else np.arange(n,len(frame))
        assert not np.isin(groups[t],groups[q]).any()
        path=OUT/f'fold{k}.npz'
        if not path.exists():
            bits,coarse,matrix=coding(fields,t)
            p=scores(bits,coarse,t,q,y)
            tree=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=17,min_samples_leaf=22,l2_regularization=4,learning_rate=.055,early_stopping=False,random_state=737)
            direct=tree.fit(matrix[t],y[t]).predict_proba(matrix[q])[:,1]
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,t=t,q=q,labels=y[t],ids=ids[q],scores=np.column_stack([p,direct]))
            tmp.replace(path)
        with np.load(path) as z:
            for key,val in [('t',t),('q',q),('labels',y[t]),('ids',ids[q])]:np.testing.assert_array_equal(z[key],val)
            probs[q]=z['scores']
        print(k,flush=True)
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        np.testing.assert_array_equal(z['f'],f)
        np.testing.assert_array_equal(z['y'],y)
        parent=z['parent']
    np.testing.assert_array_equal(parent[n:],test_decisions(NAME,pd.Series(ids[n:]))[0])
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        z_foundation=z['scores'][:,-3:]
        strength=abs(logit(np.median(z_foundation,axis=1).clip(.005,.995)))
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        first=z['champion111'].astype(bool)
    first=np.r_[first,test_decisions('proposal_n26_champion_fare_six',pd.Series(ids[n:]))[0]]
    assert (first[n:]!=parent[n:]).sum()==16
    names=['independent','context','tree']
    report=dict(standalone={k:dict(accuracy=float(np.mean((p[:n]>=.5)==y)),loss=float(log_loss(y,p[:n]))) for k,p in zip(names,probs.T)},rules={})
    delta=logit(probs[:,1].clip(.001,.999))-logit(probs[:,0].clip(.001,.999))
    votes=(probs>=.5)!=parent[:,None]
    anchor=np.where(parent,strength,-strength)
    increment=((anchor+delta)>=0)!=parent
    acts=dict(zip(names,votes.T))
    acts.update(exclusive=votes[:,1]&~votes[:,0]&~votes[:,2],increment=increment,confirmed=votes[:,1]&votes[:,2]&increment)
    # Learn the abstention threshold on different orders; exclude the 16 parent edits only on application.
    risk=np.where(first,1-probs[:,1],probs[:,1])
    for mode in ['random','ordered']:
        bins=partition(g,'manifest_ctw',4) if mode=='random' else np.zeros(n,int)
        act=np.zeros(len(ids),bool); packs=[]
        for k in range(8):
            if mode=='ordered':
                for b,grp in enumerate(np.array_split(np.unique(g[f==k]),4)):
                    bins[(f==k)&np.isin(g,grp)]=b
            for j in [0,1,2,3,None]:
                t=np.flatnonzero(known&(f==k)&((bins!=j) if j is not None else True))
                q=np.arange(n,len(ids)) if j is None else np.flatnonzero((f==k)&(bins==j))
                cut=calibrate(risk,first,y,t,1.)
                a=(risk[q]>=cut[first[q].astype(int)])&(parent[q]==first[q])
                if j is None:packs.append(a)
                else:act[q]=a
        act[n:]=np.mean(packs,axis=0)>.5
        acts['held_'+mode]=act
    # Estimate incremental value conditional on the foundation score and the entire raw-field tree.
    from scipy.optimize import minimize
    native_score=logit(np.median(z_foundation,axis=1).clip(.005,.995))
    basic=np.column_stack([np.ones(len(ids)),native_score,logit(probs[:,2].clip(.005,.995)),logit(probs[:,0].clip(.005,.995))])
    for mode in ['random','ordered']:
        bins=partition(g,'manifest_ctw',4) if mode=='random' else np.zeros(n,int)
        fitted={}
        for label,x in [('baseline',basic),('context',np.column_stack([basic,logit(probs[:,1].clip(.005,.995))]))]:
            values=np.full(len(ids),np.nan);pack=[]
            for k in range(8):
                if mode=='ordered':
                    for j,grp in enumerate(np.array_split(np.unique(g[f==k]),4)):bins[(f==k)&np.isin(g,grp)]=j
                for j in [0,1,2,3,None]:
                    t=np.flatnonzero((f==k)&((bins!=j) if j is not None else True))
                    q=np.arange(n,len(ids)) if j is None else np.flatnonzero((f==k)&(bins==j))
                    a=x[t]
                    def loss(w):
                        v=native_score[t]+a@w
                        return np.logaddexp(0,v).sum()-y[t]@v+16*w@w,a.T@(expit(v)-y[t])+32*w
                    opt=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
                    assert opt.success
                    inc=x[q]@opt.x
                    if j is None:pack.append(inc)
                    else:values[q]=inc
            values[n:]=np.mean(pack,axis=0)
            fitted[label]=values
        adv=fitted['context']-fitted['baseline']
        acts['tree_adjusted_'+mode]=((anchor+adv>=0)!=parent)&(first==parent)
    for k,act in acts.items():
        v=act[:n]*np.where(parent[:n]!=y,1,-1)
        report['rules'][k]=dict(test_changes=int(act[n:].sum()),overlap_16=int((act[n:]&(first[n:]!=parent[n:])).sum()),
                      by_fold=[int(v[known&(f==j)].sum()) for j in range(8)],
                      scopes={s:count(act[:n],m,parent[:n]!=y) for s,m in [('all',known),('fresh',known&fresh),('blind',blind)]})
    np.savez_compressed(OUT/'aligned.npz',ids=ids,fold=f,y=y,parent=parent,prob=probs,**acts)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report['standalone'],[(k,s['scopes']['all']) for k,s in report['rules'].items()])


if __name__=='__main__':
    with threadpool_limits(1):run()
