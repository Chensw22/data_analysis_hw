"""Fixed mixed categorical/zero-inflated latent classes, supervised versus unlabeled EM."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','FALSE')
import json
from pathlib import Path
import time
import signal
import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import error_overlap_diagnostic as overlap
import relational_error_corrector as r
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/latent_class_em_probe'
OUT=ROOT/'artifacts/latent_class_em_probe'
CATS=['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
BILLS=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
COLS=CATS+['Age','CabinNumber']+BILLS
K=24
ITERATIONS=80


def missing(s):return s.isna()|s.astype('string').eq('__MISSING__').fillna(False)


class Encoder:
    def fit(self,x):
        if set(x.columns)!=set(COLS):raise ValueError('Exact13 attributes only')
        self.vocab={c:{v:i for i,v in enumerate(sorted(x.loc[~missing(x[c]),c].astype(str).unique()))} for c in CATS}
        self.edges={'Age':np.array([13.,18.,30.,50.,80.])}
        for c,bins in [('CabinNumber',16)]+[(c,8) for c in BILLS]:
            a=x[c].to_numpy(float);a=a[np.isfinite(a)&((a>0) if c in BILLS else np.ones(len(a),bool))]
            self.edges[c]=np.unique(np.quantile(a,np.arange(1,bins)/bins)) if len(a) else np.array([])
        self.cards=[max(1,len(self.vocab[c])) for c in CATS]+[len(self.edges[c])+1 for c in ['Age','CabinNumber']]+[len(self.edges[c])+2 for c in BILLS]
        return self
    def transform(self,x):
        if list(x.columns)!=COLS:raise ValueError('Ordered feature schema mismatch')
        z=np.full((len(x),13),-1,dtype=int)
        for j,c in enumerate(CATS):z[:,j]=x[c].astype(str).map(self.vocab[c]).fillna(-1).to_numpy(int);z[missing(x[c]).to_numpy(),j]=-1
        for j,c in enumerate(['Age','CabinNumber']+BILLS,start=6):
            a=x[c].to_numpy(float);obs=np.isfinite(a)
            if (a[obs]<0).any():raise ValueError('Negative age/cabin/bill unsupported')
            if c in BILLS:
                z[obs&(a==0),j]=0;pos=obs&(a>0);z[pos,j]=1+np.searchsorted(self.edges[c],a[pos],side='right')
            else:z[obs,j]=np.searchsorted(self.edges[c],a[obs],side='right')
        return z


def initialize(z,cards,seed):
    onehot=np.concatenate([(z[:,j,None]==np.arange(n)[None,:]).astype(float) for j,n in enumerate(cards)],axis=1)
    labels=KMeans(n_clusters=K,n_init=1,max_iter=100,random_state=seed).fit_predict(onehot)
    return .98*np.eye(K)[labels]+.02/K


class LatentClasses:
    def __init__(self,cards):self.cards=list(cards)
    def maximize(self,z,resp,y=None):
        nk=resp.sum(0);self.pi=(nk+1)/(len(z)+K);self.tables=[]
        for j,n in enumerate(self.cards):
            counts=np.stack([resp[z[:,j]==value].sum(0) for value in range(n)],axis=1)
            if j>=8:
                # Distinct Beta(2,2) MAP pseudo-count1 for zero occurrence,
                # Dirichlet(2) MAP pseudo-count1 for positive amount bins.
                zero=counts[:,0];positive=counts[:,1:].sum(1)
                pzero=(zero+1)/(zero+positive+2)
                amounts=(counts[:,1:]+1)/(positive[:,None]+n-1)
                probs=np.column_stack([pzero,(1-pzero[:,None])*amounts])
            else:probs=(counts+1)/(counts.sum(1,keepdims=True)+n)
            self.tables.append(probs)
        if y is not None:self.py=(resp.T@y+1)/(nk+2)
    def components(self,z,labels=None):
        logp=np.broadcast_to(np.log(self.pi),(len(z),K)).copy()
        for j,table in enumerate(self.tables):
            obs=z[:,j]>=0;logp[obs]+=np.log(table[:,z[obs,j]].T)
        if labels is not None:logp+=labels[:,None]*np.log(self.py)+(1-labels[:,None])*np.log1p(-self.py)
        return logp
    def posterior(self,z):
        logp=self.components(z);return np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
    def fit(self,z,y,initial,supervised):
        resp=initial.copy();history=[]
        for step in range(ITERATIONS):
            self.maximize(z,resp,y if supervised else None)
            logp=self.components(z,y if supervised else None);ll=logsumexp(logp,axis=1)
            resp=np.exp(logp-ll[:,None]);history.append(dict(iteration=step+1,mean_observed_log_likelihood=float(ll.mean())))
        if not supervised:
            resp=self.posterior(z);self.py=(resp.T@y+1)/(resp.sum(0)+2)
        self.history=history
        if not np.isfinite(self.pi).all() or not np.isfinite(self.py).all():raise RuntimeError('Nonfinite EM')
        return self
    def predict(self,z):return self.posterior(z)@self.py


def checks():
    m=LatentClasses([2]*13);z=np.array([[0]*13,[1]*13,[-1]*13]);responsibility=np.full((3,K),1/K)
    m.maximize(z,responsibility,np.array([0,1,1]));q=m.predict(np.full((1,13),-1))
    np.testing.assert_allclose(q,m.pi@m.py,rtol=0,atol=1e-14)
    for table in m.tables:np.testing.assert_allclose(table.sum(1),1)
    attrs=pd.DataFrame({c:['A','B',None] for c in CATS})
    attrs['Age']=[12.,30.,np.nan];attrs['CabinNumber']=[1.,2.,np.nan]
    for c in BILLS:attrs[c]=[0.,10.,np.nan]
    attrs=attrs[COLS];enc=Encoder().fit(attrs);encoded=enc.transform(attrs)
    assert (encoded[0,8:]==0).all() and (encoded[1,8:]>0).all() and (encoded[2]==-1).all()
    np.testing.assert_array_equal(enc.transform(attrs.iloc[::-1]),encoded[::-1])
    return dict(missing_fields_marginalized=True,zero_positive_missing_encoding=True,permutation=True)


def run():
    if REPORT.exists() or OUT.exists():raise FileExistsError('Fixed experiment no overwrite')
    check=checks();inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y'];x=inputs['features'][COLS]
    hist=json.loads((ROOT/'reports/v96_oof_completion/result.json').read_text());path=ROOT/'artifacts/v96_oof_completion/oof.npz'
    assert hist==json.loads((ROOT/'artifacts/v96_oof_completion/result.json').read_text()) and hist['status']=='complete_v96_fixed15_original3fold_oof'
    assert r.sha(path)==hist['oof_sha256']
    with np.load(path) as z:
        for key,w in [('dev_ids',dev),('fold_id',folds),('passenger_ids',inputs['passengers'])]:np.testing.assert_array_equal(z[key],w)
        v96=z['prob'].copy()
    sources={**evidence,**{n:r.sha(ROOT/n) for n in [Path(__file__).name,'error_overlap_diagnostic.py','relational_error_corrector.py','reports/v96_oof_completion/result.json','artifacts/v96_oof_completion/result.json','artifacts/v96_oof_completion/oof.npz','artifacts/error_rescue_20261117/catalog_rows.csv']}}
    REPORT.mkdir(parents=True);OUT.mkdir(parents=True)
    plan=dict(status='fixed_before_fit',K=K,EM_iterations=ITERATIONS,seed='20261112+outer',columns=COLS,
        initialization='Shared attributes-only onehot KMeans24/n_init1/max_iter100, responsibilities .98onehot+.02/K; no restarts/selection.',
        categories='Training vocabulary multinomial with pseudo-count1; missing/unknown excluded from likelihood, not filled.',
        age='Fixed boundaries13,18,30,50,80; missing marginalized.',cabin='16 quantile bins fitted on observed outertrain only.',
        bills='Each bill: zero-vs-positive Bernoulli pseudo-count1 each; positive amount8 quantile bins outertrain only, Dirichlet pseudo-count1. Missing bill integrates entire bill likelihood to1.',
        supervision='Supervised arm observes y ONLY on outertrain as extra Bernoulli factor during EM; predict accepts attributes only and sums latent classes. Unsupervised arm EM never sees y then estimates class probability from final training responsibilities.',
        controls='Same initializer, K, schema, smoothing,80EM iterations; both solo and fixed half-v96 probabilities; no thresholds/weights/seed search.',
        split='Original6954dev SKF3seed20261112; old1739 targets never read; full test not model training.',threshold=.5,CPU_threads=4,budget_seconds=600,
        difference='Finite mixture of conditional-independent typed likelihoods with y-guided latent classes, not a single dependency tree, directed bill chain, or rule imputation.',
        limitations=['No claim of new observed information.','Fixed discretization can lose metric/spatial information; initialization/local optimum may matter.','Within-class conditional independence may remain false.','Missingness mechanism not modeled; marginalization assumes ignorable missingness.','Adaptive reuseddev; hard masks are reporting only.'])
    r.write(REPORT/'protocol.json',plan);sources[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json');r.write(REPORT/'freeze.json',sources);r.write(REPORT/'checks.json',check)
    start=time.monotonic();old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('LatentEMCPU600s')));signal.setitimer(signal.ITIMER_REAL,600)
    pred={k:np.full(len(y),np.nan) for k in ['unsupervised','supervised','unsupervised_half_v96','supervised_half_v96']};pred.update(raw=bases['rawMitra'],v96=v96);records=[]
    try:
        with threadpool_limits(limits=4):
            for fold,(a,b) in enumerate(splits):
                if np.intersect1d(dev[a],dev[b]).size:raise RuntimeError('Outer row intersection')
                enc=Encoder().fit(x.loc[dev[a]]);za=enc.transform(x.loc[dev[a]]);zb=enc.transform(x.loc[dev[b]])
                initial=initialize(za,enc.cards,20261112+fold);meta={}
                for name,supervised in [('unsupervised',False),('supervised',True)]:
                    model=LatentClasses(enc.cards).fit(za,y[a].copy(),initial,supervised)
                    p=model.predict(zb);pred[name][b]=p;pred[name+'_half_v96'][b]=.5*p+.5*v96[b]
                    path=OUT/f'fold{fold}_{name}.joblib';joblib.dump(dict(encoder=enc,model=model,columns=COLS),path);saved=joblib.load(path)
                    np.testing.assert_array_equal(p,saved['model'].predict(saved['encoder'].transform(x.loc[dev[b]])))
                    resp=model.posterior(za)
                    meta[name]=dict(model_sha256=r.sha(path),reload_max_diff=0.,history=model.history,component_weights=model.pi.tolist(),component_transport_probability=model.py.tolist(),training_attribute_posterior_mass=resp.sum(0).tolist(),metrics=r.score(p,y[b]))
                record=dict(fold=fold,models=meta,train_ids_sha256=r.nested.ids_hash(dev[a]),query_ids_sha256=r.nested.ids_hash(dev[b]),bin_cardinalities=enc.cards,
                    quantile_edges={k:v.tolist() for k,v in enc.edges.items()},metrics={k:r.score(v[b],y[b]) for k,v in pred.items()})
                records.append(record);r.write(REPORT/f'fold{fold}.json',record);print(json.dumps(dict(fold=fold,metrics=record['metrics'])),flush=True)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for key,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[key],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        xd=x.loc[dev];complete=~np.column_stack([missing(xd[c]).to_numpy() for c in COLS]).any(1)
        earth=xd.HomePlanet.astype(str).to_numpy()=='Earth';asleep=xd.CryoSleep.astype(str).str.lower().eq('true').to_numpy();awake=xd.CryoSleep.astype(str).str.lower().eq('false').to_numpy()
        masks=dict(all=np.ones(len(y),bool),hard546=hard,complete13=complete,hard546_complete13=hard&complete,missing13=~complete,Earth_sleep=earth&asleep,Earth_awake=earth&awake)
        def stat(mask):
            values={}
            for k,p in pred.items():
                yh=p>=.5;values[k]=dict(correct=int(((yh==y)&mask).sum()),vs_raw=r.pair(p,pred['raw'],y,mask),vs_v96=r.pair(p,v96,y,mask),vs_unsupervised=r.pair(p,pred['unsupervised'],y,mask),vs_unsupervised_half=r.pair(p,pred['unsupervised_half_v96'],y,mask),TN=int((mask&(y==0)&~yh).sum()),FP=int((mask&(y==0)&yh).sum()),FN=int((mask&(y==1)&~yh).sum()),TP=int((mask&(y==1)&yh).sum()))
            return dict(rows=int(mask.sum()),models=values)
        for i,record in enumerate(records):record['paired']=stat(folds==i);record['hard546']=stat(hard&(folds==i));r.write(REPORT/f'fold{i}.json',record)
        path=OUT/'oof.npz';np.savez_compressed(path,**pred,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        for name,digest in sources.items():assert r.sha(ROOT/name)==digest,name
        output=dict(status='complete',metrics={k:r.score(v,y) for k,v in pred.items()},subsets={k:stat(m) for k,m in masks.items()},folds=records,seconds=time.monotonic()-start,oof_sha256=r.sha(path),source_hashes=sources,no_upload=True,old_holdout_labels_read=False)
        r.write(REPORT/'result.json',output);print(json.dumps(output['metrics']),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
Encoder.__module__='latent_class_em_probe';LatentClasses.__module__='latent_class_em_probe'
if __name__=='__main__':
    import sys
    sys.modules['latent_class_em_probe']=sys.modules[__name__]
    run()
