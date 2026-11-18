"""One fixed nonlinear CPU readout of authenticated cross-fitted Mitra embeddings."""
import os
if os.environ.get('KMP_DUPLICATE_LIB_OK') not in (None,'FALSE'):raise RuntimeError('Fresh FALSE process required')
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import json
from pathlib import Path
import time
import signal
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import relational_error_corrector as r
import error_overlap_diagnostic as overlap
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT/'artifacts/mitra_hidden_readout_probe'
REPORT=ROOT/'reports/mitra_hidden_nystrom_probe'
OUT=ROOT/'artifacts/mitra_hidden_nystrom_probe'


def run():
    if OUT.exists() or REPORT.exists():raise FileExistsError('No overwrite or tuning')
    inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y']
    result=json.loads((ROOT/'reports/mitra_hidden_readout_probe/result.json').read_text())
    audit=json.loads((ROOT/'reports/mitra_hidden_readout_probe/audit.json').read_text())
    assert result['status']=='complete' and audit['status']=='verified_complete'
    assert r.sha(PRIOR/'oof.npz')==result['oof_sha256']==audit['oof_sha256']
    for name,digest in result['source_hashes'].items():assert r.sha(ROOT/name)==digest,name
    with np.load(PRIOR/'oof.npz',allow_pickle=False) as z:
        for key,w in [('dev_ids',dev),('fold_id',folds),('passenger_ids',inputs['passengers'])]:np.testing.assert_array_equal(z[key],w)
        predictions={k:z[k].copy() for k in ['raw','constant','hidden']}
    np.testing.assert_array_equal(predictions['raw'],bases['rawMitra'])
    evidence.update(result['source_hashes'])
    for path in list(PRIOR.glob('*.npz'))+list(PRIOR.glob('*_readout.joblib'))+[ROOT/'reports/mitra_hidden_readout_probe/result.json',ROOT/'reports/mitra_hidden_readout_probe/audit.json',ROOT/Path(__file__).name]:
        evidence[str(path.relative_to(ROOT))]=r.sha(path)
    REPORT.mkdir(parents=True);OUT.mkdir(parents=True)
    plan=dict(status='frozen_before_PCA_kernel_or_readout_fit',source='Authenticated strictnested 512-dimensional label-token vectors, raw/constant/linear512 control unchanged',
        preprocessing='Original outertrainOOF mean/std then full-SVD PCA32 fitted only outertrainOOF; outerquery transformed only.',
        landmarks='256 training rows selected by sklearn Nystroem random_state20261112+fold; RBF and linear share exact component_indices_',
        bandwidth='Median of all nonself pairwise Euclidean distances among outertrain PCA32 rows; gamma=1/(2*median**2). No query distances used.',
        kernels=['rbf','linear'],kernel_normalization='sklearn native Nyström inverse-square-root landmark kernel, singular values floor1e-12. No subsequent column scaling: preserve kernel feature geometry.',
        control='Linear-kernel Nyström with same256 landmarks; its rank<=32 and function space is linear in PCA32.',
        objective='Same fixed raw logit offset; mean binary NLL+.1||w||²; intercept free; L-BFGS-B500/ftol1e-12/gtol1e-8, no search',
        threshold=.5,CPU_threads=4,budget_seconds=600,no_GPU=True,no_base_fit=True,
        limitations=['Adaptive reuseddev, not independent validation.','RBF and linear kernel norms differ despite same penalty, so result is recipe comparison not isolated causality.','Inner versus outercontext size shift persists.','Hard546 only posthoc reporting.'])
    r.write(REPORT/'protocol.json',plan);evidence[str((REPORT/'protocol.json').relative_to(ROOT))]=r.sha(REPORT/'protocol.json')
    r.write(REPORT/'freeze.json',dict(sources=evidence));start=time.monotonic()
    def timeout(*_):raise TimeoutError('FixedCPU600s')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    predictions.update(pca_linear=np.full(len(y),np.nan),rbf=np.full(len(y),np.nan));records=[]
    try:
        with threadpool_limits(limits=4):
            for outer,(a,b) in enumerate(splits):
                h=np.empty((len(a),512));ip=np.empty(len(a));coverage=np.zeros(len(a),int)
                for j,(ia,ib) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261117+outer).split(a,y[a])):
                    path=PRIOR/f'outer{outer}_inner{j}.npz';assert r.sha(path)==result['folds'][outer]['inner_jobs'][j]['packet_sha256']
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z['train_ids'],dev[a[ia]]);np.testing.assert_array_equal(z['query_ids'],dev[a[ib]])
                        assert not np.intersect1d(z['train_ids'],z['query_ids']).size
                        assert not np.intersect1d(z['train_ids'],dev[b]).size
                        h[ib]=z['embedding'];ip[ib]=z['prob'];coverage[ib]+=1
                assert np.all(coverage==1)
                path=PRIOR/f'outer{outer}_query.npz';assert r.sha(path)==result['folds'][outer]['outer_job']['packet_sha256']
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['train_ids'],dev[a]);np.testing.assert_array_equal(z['query_ids'],dev[b]);np.testing.assert_array_equal(z['prob'],predictions['raw'][b]);hq=z['embedding'].copy()
                mean=h.mean(0);scale=h.std(0);scale[scale==0]=1.
                pca=PCA(n_components=32,svd_solver='full').fit((h-mean)/scale)
                t=pca.transform((h-mean)/scale);q=pca.transform((hq-mean)/scale)
                distances=pdist(t,metric='euclidean');median=float(np.median(distances));del distances
                if not median>0:raise RuntimeError('Degenerate fixed bandwidth')
                gamma=1/(2*median**2);fitlogs={};landmarks=None
                for name,kernel in [('pca_linear','linear'),('rbf','rbf')]:
                    mapping=Nystroem(kernel=kernel,gamma=gamma if kernel=='rbf' else None,n_components=256,random_state=20261112+outer,n_jobs=4)
                    za=mapping.fit_transform(t);zq=mapping.transform(q)
                    if landmarks is None:landmarks=mapping.component_indices_.copy()
                    else:np.testing.assert_array_equal(landmarks,mapping.component_indices_)
                    theta,fit=r.fit_corrector(ip,y[a],za)
                    predictions[name][b]=expit(r.logit(predictions['raw'][b])+theta[0]+zq@theta[1:])
                    path=OUT/f'fold{outer}_{name}.joblib';joblib.dump(dict(mean=mean,scale=scale,pca=pca,mapping=mapping,theta=theta),path)
                    loaded=joblib.load(path);lq=loaded['pca'].transform((hq-loaded['mean'])/loaded['scale']);lz=loaded['mapping'].transform(lq)
                    np.testing.assert_array_equal(predictions[name][b],expit(r.logit(predictions['raw'][b])+loaded['theta'][0]+lz@loaded['theta'][1:]))
                    fitlogs[name]=dict(**fit,model_sha256=r.sha(path),reload_max_diff=0.)
                np.savez_compressed(OUT/f'fold{outer}.npz',train_ids=dev[a],query_ids=dev[b],landmark_ids=dev[a[landmarks]],pca_linear=predictions['pca_linear'][b],rbf=predictions['rbf'][b])
                record=dict(fold=outer,bandwidth_median=median,gamma=gamma,pca_variance_fraction=float(pca.explained_variance_ratio_.sum()),fits=fitlogs,
                    metrics={k:r.score(v[b],y[b]) for k,v in predictions.items()},paired={k:r.pair(predictions['rbf'][b],predictions[k][b],y[b]) for k in ['raw','constant','pca_linear','hidden']})
                records.append(record);r.write(REPORT/f'fold{outer}.json',record);print(json.dumps(dict(fold=outer,metrics=record['metrics'],paired=record['paired'])),flush=True)
        rows=pd.read_csv(ROOT/'artifacts/error_rescue_20261117/catalog_rows.csv',dtype={'PassengerId':str})
        for key,w in [('row_id',dev),('y',y),('fold',folds),('PassengerId',inputs['passengers'])]:np.testing.assert_array_equal(rows[key],w)
        hard=rows.all_recipes_still_wrong.to_numpy(bool);assert hard.sum()==546
        def summary(m):return {k:dict(correct=int((((v>=.5)==y)&m).sum()),vs_raw=r.pair(v,predictions['raw'],y,m),vs_constant=r.pair(v,predictions['constant'],y,m)) for k,v in predictions.items()}
        for i,record in enumerate(records):record['hard546']=summary(hard&(folds==i));r.write(REPORT/f'fold{i}.json',record)
        path=OUT/'oof.npz';np.savez_compressed(path,**predictions,dev_ids=dev,fold_id=folds,passenger_ids=np.asarray(inputs['passengers'],dtype=str))
        for name,digest in evidence.items():assert r.sha(ROOT/name)==digest,name
        output=dict(status='complete',metrics={k:r.score(v,y) for k,v in predictions.items()},subsets=dict(all=summary(np.ones(len(y),bool)),hard546=summary(hard)),folds=records,seconds=time.monotonic()-start,
            oof_sha256=r.sha(path),source_hashes=evidence,no_gpu=True,no_base_refit=True,no_upload=True)
        r.write(REPORT/'result.json',output);r.write(OUT/'result.json',output);print(json.dumps(output['metrics']),flush=True)
    except BaseException as exc:r.write(REPORT/'failure.json',dict(type=type(exc).__name__,error=str(exc)));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
if __name__=='__main__':run()
