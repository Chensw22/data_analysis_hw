"""Exploratory query-group structure only; no model/coupling fit or candidate score."""
import json, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import audit_top5_references as audit
OUT=Path(__file__).resolve().parent

def bootstrap(rows, numerator, denominator):
    rng=np.random.default_rng(20261116);values=[]
    for _ in range(2000):
        sampled=rows[rng.integers(len(rows),size=len(rows))]
        values.append(sampled[:,numerator].sum()/sampled[:,denominator].sum())
    return np.quantile(values,[.025,.975]).tolist()

def run():
    path=ROOT/'artifacts/run_002/split_indices.npz'
    dev=np.sort(np.load(path)['development']).astype(np.int64)
    assert len(dev)==6954 and audit.row_hash(dev)==audit.DEV_HASH
    train,test,y=audit.read_development_only(ROOT/'data/raw/train.csv',ROOT/'data/raw/test.csv',dev)
    groups=train.PassengerId.iloc[dev].str.split('_').str[0].to_numpy()
    _,inverse=np.unique(groups,return_inverse=True)
    members=[np.flatnonzero(inverse==i) for i in range(inverse.max()+1)]
    multiple=[a for a in members if len(a)>1]
    result=dict(status='exploratory_only_no_model_or_coupling_fit',development_ids_sha256=audit.row_hash(dev),
        labels_read='original6954development_only',source_sha256={},group_sizes={},references={},
        bootstrap='2000 resamples of entire multimember groups, seed20261116; descriptive CI, no adaptive-selection correction')
    for name,frame in [('development',train.iloc[dev]),('full_train_attributes',train),('test_attributes',test)]:
        sizes=frame.PassengerId.str.split('_').str[0].value_counts()
        result['group_sizes'][name]=dict(rows=len(frame),groups=len(sizes),multimember_groups=int((sizes>1).sum()),
            multimember_rows=int(sizes[sizes>1].sum()),max_size=int(sizes.max()),
            group_size_frequency={str(k):int(v) for k,v in sizes.value_counts().sort_index().items()})
    full_sizes=train.PassengerId.str.split('_').str[0].value_counts()
    dev_sizes=pd.Series(groups).value_counts()
    incomplete=dev_sizes[dev_sizes<full_sizes.reindex(dev_sizes.index)]
    result['development_graph_thinning']=dict(groups_missing_known_roster_members=len(incomplete),
        observed_dev_rows_in_incomplete_groups=int(incomplete.sum()),
        note='Missing members have attributes available but old holdout labels were not read or used; no graph completion performed.')
    files=['audit_top5_references.py','reports/group_residual_diagnostic.json','reports/relational_review/diagnostics.json',str(path.relative_to(ROOT))]
    for reference in ['mitra','v44']:
        directory=ROOT/'artifacts/top5_reference_audit'/reference
        summary=json.loads((directory/'summary.json').read_text())
        entry=next(x for x in summary['audits'] if x['kind']=='group' and x['seed']==20261112)
        path=ROOT/entry['oof_path'];assert audit.file_hash(path)==entry['oof_sha256']
        z=np.load(path)
        assert np.array_equal(z['dev_ids'],dev) and np.array_equal(z['group_ids'].astype(str),groups)
        assert np.array_equal(z['passenger_ids'].astype(str),train.PassengerId.iloc[dev].astype(str))
        p=z['prob'];residual=y-p;rows=[];byfold=[]
        for a in multiple:
            n=len(a);r=residual[a];pp=p[a];yy=y[a]
            assert len(set(z['fold_id'][a]))==1
            pairs=n*(n-1)/2
            cross=(r.sum()**2-np.square(r).sum())/2
            observed=(yy.sum()*(yy.sum()-1)+(n-yy.sum())*(n-yy.sum()-1))/2
            expected=0.
            for j in range(n):
                for k in range(j):expected+=pp[j]*pp[k]+(1-pp[j])*(1-pp[k])
            # At beta=0 in normalized Ising, d p_i / d beta=2*p_i*(1-p_i)*mean(peer spins).
            # Thus derivative of summed marginal binary log likelihood is sum2*r_i*mean(peer spins).
            peer_spin=(np.sum(2*pp-1)-(2*pp-1))/(n-1)
            marginal_ll_gradient=float(np.sum(2*r*peer_spin))
            rows.append([pairs,cross,observed,expected,n,marginal_ll_gradient,int(z['fold_id'][a][0])])
        rows=np.asarray(rows)
        for fold in range(3):
            a=rows[rows[:,-1]==fold]
            byfold.append(dict(fold=fold,groups=len(a),pairs=int(a[:,0].sum()),
                residual_product=float(a[:,1].sum()/a[:,0].sum()),
                normalized_positive_Ising_marginal_ll_gradient_per_multimember_row=float(a[:,5].sum()/a[:,4].sum())))
        result['references'][reference]=dict(source='frozen wholeGroupCV20261112 OOF',
            multimember_groups=len(rows),pairs=int(rows[:,0].sum()),
            raw_label_agreement=float(rows[:,2].sum()/rows[:,0].sum()),
            independent_probability_expected_agreement=float(rows[:,3].sum()/rows[:,0].sum()),
            residual_product=float(rows[:,1].sum()/rows[:,0].sum()),
            residual_product_group_bootstrap95=bootstrap(rows,1,0),
            normalized_positive_Ising_marginal_ll_gradient_per_multimember_row=float(rows[:,5].sum()/rows[:,4].sum()),
            marginal_ll_gradient_group_bootstrap95=bootstrap(rows,5,4),by_fold=byfold)
        files += [str(path.relative_to(ROOT)),str((directory/'summary.json').relative_to(ROOT))]
    files.append(str(Path(__file__).relative_to(ROOT)))
    result['source_sha256']={f:audit.file_hash(ROOT/f) for f in files}
    result['interpretation_limits']=['Previously reused development labels and OOF are only descriptive here, never nested meta-training data.','Pair dependence alone does not improve per-row accuracy when marginal probabilities are held fixed.','The beta=0 gradient is an analytic descriptive direction for marginal logloss, not fitted beta, resulting accuracy, or model selection evidence.','No v85 artifacts or incomplete fold scores were inspected.']
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['group_sizes','references']},indent=2))

if __name__=='__main__':run()
