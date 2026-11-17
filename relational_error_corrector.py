"""One fixed relational logit correction using authenticated nested cached predictions."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import error_overlap_diagnostic as overlap
import planet_mixture_experiment as nested

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/error_rescue_20261117'
OUT = ROOT / 'artifacts/error_rescue_20261117'
PLAN = REPORT / 'relational_protocol.json'
FEATURES = ['log_group_peers', 'log_cabin_peers', 'cryo_x_log_group_peers',
            'cryo_x_log_cabin_peers', 'cabin_count_missing']
EPS = 1e-7
PENALTY = .1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze():
    if PLAN.exists():
        raise FileExistsError('One fixed experiment; do not overwrite existing protocol.')
    plan = dict(status='frozen_before_new_corrector_fit', features=FEATURES,
        definition='Full unlabeled roster counts. g=log1p(GroupSize-1), h=log1p(CabinOccupants-1); missing cabin h=0. Cryo c=True:+1/False:-1/missing:0. Features g,h,c*g,c*h,missingCabin.',
        preprocessing='Each outertrain computes its own feature mean/std; zero std becomes1. Query transformed with these constants.',
        base='rawMitra run038 original same three outer folds', inner='Only original planet_mixture jobs for the matching outerfold; original inner3 seeds20261117+outer.',
        objective='mean(logaddexp(0, offset+intercept+Z@beta)-y*(offset+intercept+Z@beta)) + .1*sum(beta**2)',
        offset='logit(clip(p,1e-7,1-1e-7)); coefficient fixed1', control='Same offset, fitted intercept only; no penalty on intercept.',
        optimizer=dict(method='L-BFGS-B', maxiter=500, ftol=1e-12, gtol=1e-8, initial='all zero'),
        threshold=.5, outer_seed=20261112, folds=3, old_holdout_targets_read=False,
        training_population='All outertrain rows; never select914commonwrong or any outcome-based subset.',
        diagnostic_subsets='Five-family914commonwrong and5263commoncorrect are posthoc reporting only; no fitting/selection use.',
        continuation='No new pass threshold or automatic upload. Report all folds, repairs and regressions; root interprets the single fixed experiment.',
        limitations=['Adaptive reused development evidence.', 'Inner base support about3090 vs outer4636; constant control does not remove all distribution differences.', 'Public roster attributes are transductive and contain no target propagation.'],
        sources_sha256={name:sha(ROOT/name) for name in ['relational_error_corrector.py', 'error_overlap_diagnostic.py', 'planet_mixture_experiment.py', 'exaone_validation_data.py']})
    write(PLAN, plan)
    return plan


def structural_features():
    cols = ['PassengerId', 'Cabin', 'CryoSleep']
    roster = pd.concat([pd.read_csv(ROOT/'data/raw'/file, usecols=cols)
                        for file in ['train.csv', 'test.csv']], ignore_index=True)
    group = roster.PassengerId.str.split('_').str[0]
    size = group.map(group.value_counts()).to_numpy(float)
    cabin = roster.Cabin.map(roster.Cabin.value_counts()).to_numpy(float)
    missing = ~np.isfinite(cabin)
    g = np.log1p(size-1)
    h = np.zeros(len(roster)); h[~missing] = np.log1p(cabin[~missing]-1)
    c = roster.CryoSleep.map({True:1., False:-1.}).fillna(0.).to_numpy(float)
    return np.column_stack([g, h, c*g, c*h, missing.astype(float)])


def load_inner(outer, a, b, dev, y, evidence):
    old = ROOT/'artifacts/planet_mixture_20261117'
    pub = ROOT/'reports/planet_mixture_20261117'
    row = nested.read(old/f'outer{outer}.json')
    assert sha(old/f'outer{outer}.json') == sha(pub/f'outer{outer}.json')
    assert row['outer_train_ids_sha256'] == nested.ids_hash(dev[a])
    assert row['outer_validation_ids_sha256'] == nested.ids_hash(dev[b])
    p = np.full(len(a), np.nan); coverage = np.zeros(len(a), dtype=int)
    inner = StratifiedKFold(3, shuffle=True, random_state=20261117+outer)
    for fold, (ia, ib) in enumerate(inner.split(a, y[a])):
        folder = old/'jobs'/f'outer{outer}_inner{fold}_raw'
        public = pub/'jobs'/folder.name
        for name in ['job.json', 'worker_result.json', 'metadata.json']:
            assert sha(folder/name) == sha(public/name)
            evidence[str((folder/name).relative_to(ROOT))] = sha(folder/name)
        job = nested.read(folder/'job.json'); record = nested.read(folder/'worker_result.json')
        assert job['kind'] == 'raw' and job['outer_fold'] == outer and job['inner_fold'] == fold
        assert job['seed'] == 20261117+100*outer+fold
        assert sha(folder/'rows.npz') == job['rows_sha256']
        with np.load(folder/'rows.npz', allow_pickle=False) as packet:
            assert set(packet.files) == {'train_ids','query_ids','outer_train_ids','outer_validation_ids','train_labels'}
            for key, expected in [('train_ids',dev[a[ia]]),('query_ids',dev[a[ib]]),
                                  ('outer_train_ids',dev[a]),('outer_validation_ids',dev[b]),('train_labels',y[a[ia]])]:
                np.testing.assert_array_equal(packet[key], expected)
        assert not np.intersect1d(dev[b], np.r_[dev[a[ia]],dev[a[ib]]]).size
        q = nested.verify_prediction(folder, record, job, dev[a[ia]], dev[a[ib]], y[a[ia]])
        meta = nested.read(folder/'metadata.json')
        assert meta['recipe']=='zero-shot' and meta['selected_steps']==0 and meta['seed']==job['seed']
        assert meta['training_rows']==len(ia) and meta['feature_count']==13
        p[ib] = q; coverage[ib] += 1
        for name in ['rows.npz','prediction.npz']:
            evidence[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    assert np.all(coverage==1) and np.isfinite(p).all()
    cache = old/f'outer{outer}_inner_oof.npz'
    assert sha(cache)==row['inner_oof_sha256']
    with np.load(cache, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['outer_train_ids'], dev[a])
        np.testing.assert_array_equal(saved['raw'], p)
    evidence[str(cache.relative_to(ROOT))] = sha(cache)
    return p


def logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p)-np.log1p(-p)


def fit_corrector(p, y, z):
    offset = logit(p)
    def objective(w):
        eta = offset+w[0]+z@w[1:]
        residual = expit(eta)-y
        value = np.mean(np.logaddexp(0, eta)-y*eta)+PENALTY*np.sum(w[1:]**2)
        gradient = np.r_[residual.mean(), z.T@residual/len(y)+2*PENALTY*w[1:]]
        return value, gradient
    result = minimize(objective, np.zeros(z.shape[1]+1), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Fixed optimizer failed: '+str(result.message))
    return result.x, dict(intercept=float(result.x[0]), standardized_coefficients=result.x[1:].tolist(),
                         objective=float(result.fun), gradient_max=float(np.abs(result.jac).max()),
                         iterations=int(result.nit), success=True)


def score(p,y):
    return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])))


def pair(p, reference, y, mask=None):
    mask = np.ones(len(y),bool) if mask is None else mask
    a=(p[mask]>=.5)==y[mask]; b=(reference[mask]>=.5)==y[mask]
    return dict(rows=int(mask.sum()), repaired=int(np.sum(a&~b)), broken=int(np.sum(~a&b)),
                net_correct=int(np.sum(a)-np.sum(b)), both_correct=int(np.sum(a&b)),both_wrong=int(np.sum(~a&~b)))


def run():
    plan=freeze(); start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    result_path=REPORT/'relational_result.json'
    try:
        inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated()
        dev,y=inputs['dev'],inputs['y']; raw=bases['rawMitra']
        x=structural_features()[dev]
        assert x.shape==(6954,5) and np.isfinite(x).all()
        # Authenticate cached execution source identity without running its environment/fit gates.
        frozen=nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        nested.verify_hashes(frozen['sources'])
        predictions={'raw':raw.copy(),'constant':np.full(len(dev),np.nan),'relational':np.full(len(dev),np.nan)}
        folds=[]
        for outer,(a,b) in enumerate(splits):
            inner=load_inner(outer,a,b,dev,y,evidence)
            mean=x[a].mean(axis=0); scale=x[a].std(axis=0); scale[scale==0]=1.
            za=(x[a]-mean)/scale; zb=(x[b]-mean)/scale
            constant,constant_fit=fit_corrector(inner,y[a],np.empty((len(a),0)))
            coeff,relation_fit=fit_corrector(inner,y[a],za)
            predictions['constant'][b]=expit(logit(raw[b])+constant[0])
            predictions['relational'][b]=expit(logit(raw[b])+coeff[0]+zb@coeff[1:])
            fold=dict(fold=outer,train_ids_sha256=nested.ids_hash(dev[a]),query_ids_sha256=nested.ids_hash(dev[b]),
                      scaling_mean=mean.tolist(),scaling_std=scale.tolist(),constant=constant_fit,relational=relation_fit,
                      metrics={k:score(p[b],y[b]) for k,p in predictions.items()},
                      paired={k:pair(predictions['relational'][b],predictions[k][b],y[b]) for k in ['raw','constant']})
            folds.append(fold); write(REPORT/f'relational_fold{outer}.json',fold)
            print(json.dumps(fold),flush=True)
        assert all(np.isfinite(p).all() for p in predictions.values())
        # Only now construct outcome-based reporting subsets, never inputs to either fit.
        wrong=np.column_stack([(bases[k]>=.5)!=y for k in overlap.FAMILIES])
        common_wrong=wrong.all(axis=1);common_correct=(~wrong).all(axis=1)
        assert int(common_wrong.sum())==914 and int(common_correct.sum())==5263
        subsets={}
        for label,mask in [('five_family_common_wrong',common_wrong),('five_family_common_correct',common_correct)]:
            subsets[label]=dict(rows=int(mask.sum()),correct_by_model={k:int(np.sum((p[mask]>=.5)==y[mask])) for k,p in predictions.items()},
                               paired={k:pair(predictions['relational'],predictions[k],y,mask) for k in ['raw','constant']})
        npz=OUT/'relational_oof.npz'
        np.savez(npz,dev_ids=dev,fold_id=fold_ids,passenger_ids=inputs['passengers'],group_ids=inputs['groups'],
                 **predictions,common_wrong=common_wrong,common_correct=common_correct)
        for name,digest in {**plan['sources_sha256'],**evidence}.items():
            assert sha(ROOT/name)==digest, name
        result=dict(status='complete',protocol_sha256=sha(PLAN),folds=folds,
                    metrics={k:score(p,y) for k,p in predictions.items()},
                    paired={k:pair(predictions['relational'],predictions[k],y) for k in ['raw','constant']},
                    diagnostic_subsets=subsets,seconds=time.monotonic()-start,
                    oof_path=str(npz.relative_to(ROOT)),oof_sha256=sha(npz),
                    no_new_base_model_fits=True,no_old_holdout_targets=True,no_upload=True,
                    disclosure='One fixed adaptive-development relational correction; all training rows, strict cached innerOOF. No subset fitting or new continuation gate.')
        write(REPORT/'relational_evidence.json',evidence);write(result_path,result)
        return result
    except BaseException as error:
        write(REPORT/'relational_failure.json',dict(status='failed_inconclusive',error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
