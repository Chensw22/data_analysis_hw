"""One scalar query-group marginal model; strict nested cached unary predictions."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold

import relational_error_corrector as relational
import error_overlap_diagnostic as overlap

ROOT = Path(__file__).resolve().parent
PLAN = ROOT/'reports/error_targeted_20261118/query_group_protocol.json'
REPORT = ROOT/'reports/error_targeted_20261118/query_group_result.json'
OUT = ROOT/'artifacts/error_targeted_20261118/query_group'
PENALTY, EPS = .1, 1e-7
sha, write = relational.sha, relational.write


class QueryGraphs:
    def __init__(self, p, groups, partitions=None):
        self.p = np.asarray(p).copy()
        groups = np.asarray(groups).astype(str)
        partitions = np.zeros(len(p), dtype=int) if partitions is None else np.asarray(partitions)
        if (self.p.ndim != 1 or groups.shape != self.p.shape or partitions.shape != self.p.shape
                or not np.isfinite(p).all() or np.any((self.p<0)|(self.p>1))):
            raise ValueError('Aligned finite unary probabilities and group/partition IDs required.')
        index = {}
        for i,key in enumerate(zip(partitions.tolist(),groups.tolist())):
            index.setdefault(key,[]).append(i)
        by_size = {}
        self.covered = np.zeros(len(p),bool)
        for ids in index.values():
            if len(ids)>8:raise ValueError('Frozen graph maximum8 exceeded.')
            if len(ids)>1:
                by_size.setdefault(len(ids),[]).append(ids);self.covered[ids]=True
        self.group_size_counts={str(n):sum(len(v)==n for v in index.values()) for n in sorted({len(v) for v in index.values()})}
        self.blocks=[]
        clipped=np.clip(self.p.astype(float),EPS,1-EPS)
        logits=np.log(clipped)-np.log1p(-clipped)
        for n,ids in by_size.items():
            ix=np.asarray(ids,dtype=int)
            states=((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(float)
            spins=2*states-1
            coupling=(spins.sum(axis=1)**2-n)/(2*(n-1))
            self.blocks.append((ix,states,logits[ix]@states.T,coupling))

    def marginal(self,beta):
        if not 0<=beta<=1:raise ValueError('Frozen beta range[0,1].')
        if beta==0:return self.p.copy()  # Exact baseline, including 0/1 endpoints.
        out=self.p.astype(float,copy=True)
        for ix,states,unary,coupling in self.blocks:
            energy=unary+beta*coupling[None,:]
            weights=np.exp(energy-logsumexp(energy,axis=1,keepdims=True))
            out[ix]=(weights@states)
        return out


def predict(p,groups,beta,partitions=None):
    graph=QueryGraphs(p,groups,partitions)
    # Preserve original probability dtype and singleton values exactly at inference.
    result=graph.marginal(beta).astype(np.asarray(p).dtype,copy=False)
    if not np.array_equal(result[~graph.covered],np.asarray(p)[~graph.covered]):
        raise RuntimeError('Singleton changed.')
    return result,graph.covered


def fit_beta(p,y,groups,partitions):
    y=np.asarray(y)
    if y.shape!=np.asarray(p).shape or not np.isin(y,[0,1]).all():raise ValueError('Aligned training labels required.')
    graph=QueryGraphs(p,groups,partitions)
    def objective(beta):
        q=np.clip(graph.marginal(float(beta)).astype(float),EPS,1-EPS)
        return float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))+PENALTY*beta**2)
    result=minimize_scalar(objective,bounds=(0.,1.),method='bounded',options={'xatol':1e-5})
    if not result.success or not np.isfinite(result.fun):raise RuntimeError('Fixed scalar optimizer failed.')
    # Exact objective tie prefers lower beta; endpoints always considered.
    candidates=[(objective(v),v) for v in [0.,float(result.x),1.]]
    value,beta=min(candidates)
    return dict(beta=beta,objective=value,baseline_objective=candidates[0][0],
        optimizer_success=True,nfev=int(result.nfev),endpoint_objectives=[candidates[0][0],candidates[-1][0]],
        training_rows=len(y),covered_training_rows=int(graph.covered.sum()),
        covered_training_fraction=float(graph.covered.mean()),training_query_graph_size_counts=graph.group_size_counts,penalty=PENALTY)


def check():
    p=np.array([.2,.7,.4,.8,.35],dtype=np.float32);g=np.array(['a','a','b','b','s'])
    q,_=predict(p,g,0.)
    assert q.dtype==p.dtype and q.tobytes()==p.tobytes()
    q,_=predict(p,g,.3);assert q[-1].tobytes()==p[-1].tobytes()
    order=np.array([4,1,3,0,2]);permuted,_=predict(p[order],g[order],.3)
    np.testing.assert_array_equal(q[order],permuted)
    changed=p.copy();changed[2:4]=[.01,.02]
    np.testing.assert_array_equal(q[:2],predict(changed,g,.3)[0][:2])
    split=np.array([0,1,0,1,0]);isolated,_=predict(p,g,.3,split)
    np.testing.assert_array_equal(isolated,p)
    a,b=map(float,p[:2]);beta=.3
    # Independent closed-form four-state calculation: signs equal receive exp(beta).
    w=np.array([(1-a)*(1-b)*np.exp(beta),(1-a)*b*np.exp(-beta),
                a*(1-b)*np.exp(-beta),a*b*np.exp(beta)])
    expected=np.array([(w[2]+w[3])/w.sum(),(w[1]+w[3])/w.sum()],dtype=p.dtype)
    np.testing.assert_allclose(q[:2],expected,rtol=0,atol=3e-8)
    packed=json.loads(json.dumps({'beta':.3}))
    np.testing.assert_array_equal(q,predict(p,g,packed['beta'])[0])
    print('Synthetic beta0/singleton/permutation/group-isolation/partition-isolation/two-person enumeration/JSON reload passed.')


def run():
    if OUT.exists() or REPORT.exists():raise RuntimeError('One frozen experiment; no overwrite/resume.')
    plan=json.loads(PLAN.read_text())
    if plan['penalty']!=PENALTY or plan['epsilon']!=EPS or plan['beta_bounds']!=[0,1]:raise RuntimeError('Fixed objective differs.')
    names=[Path(__file__).name,str(PLAN.relative_to(ROOT)),*plan['source_dependencies']]
    tracked=set(filter(None,subprocess.check_output(['git','ls-files','-z','--',*names],cwd=ROOT).decode().split('\0')))
    if tracked!=set(names) or subprocess.check_output(['git','status','--porcelain','--',*names],cwd=ROOT).strip():
        raise RuntimeError('Commit protocol and source dependencies before fitting.')
    for path,digest in plan['pinned_evidence_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Pinned evidence changed: '+path)
    OUT.mkdir(parents=True);started=time.monotonic()
    def timeout(*_):raise TimeoutError('Complete CPU experiment exceeded120seconds.')
    handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        inputs,splits,fold_ids,bases,evidence=overlap.load_authenticated()
        dev,y,groups=inputs['dev'],inputs['y'],inputs['groups'];raw=bases['rawMitra']
        frozen=relational.nested.read(ROOT/'artifacts/planet_mixture_20261117/freeze.json')
        relational.nested.verify_hashes(frozen['sources'])
        evidence.update(frozen['sources']);evidence.update(plan['pinned_evidence_sha256'])
        evidence.update({n:sha(ROOT/n) for n in names})
        pred=np.full(raw.shape,np.nan,dtype=raw.dtype);covered=np.zeros(len(y),bool);folds=[]
        # Authenticate all nine packets and freeze evidence before the first scalar fit.
        prepared=[]
        for outer,(a,b) in enumerate(splits):
            inner=relational.load_inner(outer,a,b,dev,y,evidence)
            partition=np.full(len(a),-1,dtype=int)
            for fold,(_,ib) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261117+outer).split(a,y[a])):
                partition[ib]=fold
            assert np.all(partition>=0)
            for prefix in ('artifacts','reports'):
                path=f'{prefix}/planet_mixture_20261117/outer{outer}.json';evidence[path]=sha(ROOT/path)
            prepared.append((inner,partition))
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write(OUT/'freeze.json',dict(source_commit=commit,protocol=plan,evidence_sha256=evidence))
        for outer,((a,b),(inner,partition)) in enumerate(zip(splits,prepared)):
            # Only outertraining y goes into fitting. Error masks are not loaded yet.
            fit=fit_beta(inner,y[a],groups[a],partition)
            pack=dict(beta=fit['beta'],energy=plan['energy'],seed=20261117+outer,
                outer_train_ids_sha256=relational.nested.ids_hash(dev[a]),
                query_ids_sha256=relational.nested.ids_hash(dev[b]),fit=fit)
            path=OUT/f'fold{outer}_model.json';write(path,pack)
            p,c=predict(raw[b],groups[b],fit['beta'])
            restored=json.loads(path.read_text())
            reload_p,reload_c=predict(raw[b],groups[b],restored['beta'])
            np.testing.assert_array_equal(p,reload_p);np.testing.assert_array_equal(c,reload_c)
            pred[b]=p;covered[b]=c
            folds.append(dict(fold=outer,fit=fit,model_sha256=sha(path),reload_difference=0.,
                metrics=relational.score(p,y[b]),raw_metrics=relational.score(raw[b],y[b]),
                paired=relational.pair(p,raw[b],y[b]),covered_query_rows=int(c.sum()),
                covered_query_fraction=float(c.mean()),query_graph_size_counts=QueryGraphs(raw[b],groups[b]).group_size_counts))
        if not np.isfinite(pred).all():raise RuntimeError('Incomplete candidate.')
        # Outcome-defined catalog is first parsed after every fold prediction is complete.
        rows=pd.read_csv(ROOT/plan['catalog_rows'],dtype={'PassengerId':str})
        for col,expected in [('row_id',dev),('y',y),('fold',fold_ids),('PassengerId',inputs['passengers'])]:
            np.testing.assert_array_equal(rows[col],expected)
        masks={'all':np.ones(len(y),bool),'all24_wrong546':rows.all_recipes_still_wrong.to_numpy(bool),
            'primary_wrong616':rows.all_primary_still_wrong.to_numpy(bool),'five_wrong914':rows.family_wrong_count.to_numpy()==5}
        for key,n in [('all24_wrong546',546),('primary_wrong616',616),('five_wrong914',914)]:
            assert masks[key].sum()==n
        frame=inputs['features'].loc[dev]
        for state in ('True','False'):
            masks['Earth_Cryo_'+state]=(frame.HomePlanet.eq('Earth')&frame.CryoSleep.eq(state)).to_numpy()
        subsets={}
        for name,m in masks.items():
            subsets[name]=dict(**relational.pair(pred,raw,y,m),covered_rows=int((m&covered).sum()),
                candidate_correct=int(((pred>=.5)==y)[m].sum()),raw_correct=int(((raw>=.5)==y)[m].sum()),
                candidate_TN=int((m&(y==0)&(pred<.5)).sum()),candidate_TP=int((m&(y==1)&(pred>=.5)).sum()),
                candidate_FP=int((m&(y==0)&(pred>=.5)).sum()),candidate_FN=int((m&(y==1)&(pred<.5)).sum()),
                raw_TN=int((m&(y==0)&(raw<.5)).sum()),raw_TP=int((m&(y==1)&(raw>=.5)).sum()),
                raw_FP=int((m&(y==0)&(raw>=.5)).sum()),raw_FN=int((m&(y==1)&(raw<.5)).sum()))
        np.savez(OUT/'oof.npz',prob=pred,raw=raw,dev_ids=dev,fold_id=fold_ids,group_ids=groups,
            passenger_ids=inputs['passengers'],covered=covered,**masks)
        for path,digest in evidence.items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Source/evidence changed: '+path)
        result=dict(status='complete_adaptive_development_experiment',folds=folds,
            metrics=relational.score(pred,y),raw_metrics=relational.score(raw,y),subsets=subsets,
            oof_sha256=sha(OUT/'oof.npz'),freeze_sha256=sha(OUT/'freeze.json'),seconds=time.monotonic()-started,
            source_commit=commit,no_new_base_fits=True,no_old_holdout_labels=True,no_upload=True,
            interpretation=plan['limitations'])
        write(REPORT,result);print(json.dumps(result,indent=2))
    except BaseException as error:
        write(OUT/'failure.json',dict(status='failed_incomplete',error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();check() if args.check else run()
