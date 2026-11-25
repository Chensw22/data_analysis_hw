"""Test nonlinear readouts on the frozen model's actual inputs and feature-token projections."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
STORE = ROOT/'artifacts/nov25_mitra_fp32_head'
OUT = ROOT/'artifacts/nov25_mitra_token_trees'
METHODS = ('raw','tokens','joint')


def encode_inputs():
    from broad_error_transport_mitra_fp32_head_n25 import specs,load
    entries,train,ids,_ = specs()
    OUT.mkdir(exist_ok=True)
    for entry in entries:
        tag,_,digest,_,q,_,_ = entry
        model,frame = load(entry,train,ids)
        x = model.preprocessor.transform(frame)
        assert x.shape==(len(q),15) and np.isfinite(x).all()
        with np.load(STORE/f'{tag}.npz') as z:
            np.testing.assert_array_equal(z['row'],q)
            np.testing.assert_array_equal(z['passenger'],ids.iloc[q])
            np.testing.assert_array_equal(z['feature_names'],model.preprocessor.active_columns)
            assert str(z['model_sha256'])==digest
        path = OUT/f'{tag}_input.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,row=q,passenger=ids.iloc[q].to_numpy(str),raw=x,model_sha256=digest,
                                feature_names=np.array(model.preprocessor.active_columns))
        path.with_suffix('.tmp').replace(path)
        del model
        gc.collect()


def read_views(tag):
    with np.load(OUT/f'{tag}_input.npz') as a,np.load(STORE/f'{tag}.npz') as b:
        for key in ('row','passenger','model_sha256','feature_names'):
            np.testing.assert_array_equal(a[key],b[key])
        raw,token = a['raw'],b['feature_projection']
        assert np.isfinite(raw).all() and np.isfinite(token).all() and raw.shape==token.shape
        return a['row'],a['passenger'],b['native'],(raw,token,np.column_stack([raw,token]))


def fit(train,labels,queries):
    fitted = HistGradientBoostingClassifier(max_iter=115,learning_rate=.06,max_leaf_nodes=11,
            max_depth=4,min_samples_leaf=32,l2_regularization=3,early_stopping=False,random_state=784)
    fitted.fit(train,labels)
    return [fitted.predict_proba(q)[:,1] for q in queries]


def decisions(scores,base,v131,native):
    free = ((native>=.5)==v131)&(v131==base)
    results = {}
    for j,key in enumerate(METHODS):
        results[key+'_direct'] = np.where(free,scores[:,j]>=.5,base)
        results[key+'_quarter'] = np.where(free,.75*native+.25*scores[:,j]>=.5,base)
    # Apply only the difference attributable to adding token features to the same tree learner.
    delta = logit(scores[:,2].clip(.002,.998))-logit(scores[:,0].clip(.002,.998))
    results['token_increment'] = np.where(free,logit(native.clip(.002,.998))+delta>=0,base)
    return results


def run():
    from export_budget_quarter_n23 import source
    from full_roster_conditional_pool_n23 import data
    from full_roster_disagreement_calibration_n24 import stats
    from full_roster_six_failure_archive_n24 import prediction_sets
    from kaggle_workflow import sha256
    raw,y,g,f = data()
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        rows,fold,nine,v131 = [z[k] for k in ('rows','fold','nine','v131')]
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows])
        np.testing.assert_array_equal(z['y'],y[rows])
    _,panels = prediction_sets(raw,y,g,f,rows,fold)
    votes = np.column_stack(panels[1])
    consensus = ((votes>=.5)==nine[:,None]).all(1)
    assert (consensus&(nine!=y[rows])).sum()==208
    ids,test_ids,tnative,test_views = read_views('test')
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    np.testing.assert_array_equal(ids,np.arange(len(raw),len(raw)+len(test)))
    np.testing.assert_array_equal(test_ids,test.PassengerId)
    _,tv,_ = source('proposal_v131_party_consensus',test)
    _,tn,_ = source('proposal_n24_selected_dissent',test)
    hist = np.full((len(rows),len(METHODS)),np.nan)
    native = np.empty(len(rows))
    test_probs = []
    for outer in np.unique(fold):
        where = np.flatnonzero(fold==outer)
        query_ids,passenger,low,views = read_views(f'fold{outer}')
        np.testing.assert_array_equal(query_ids,rows[where])
        np.testing.assert_array_equal(passenger,raw.PassengerId.iloc[query_ids])
        native[where] = low
        free = ((low>=.5)==v131[where])&(v131[where]==nine[where])
        # Each tree sees only other bookings in this same backbone's original query fold.
        batch = []
        for a,b in GroupKFold(4,shuffle=True,random_state=128).split(where,groups=g[query_ids]):
            assert not np.isin(g[query_ids[a]],g[query_ids[b]]).any()
            learn = a[free[a]]
            batch_pred = []
            for j,v in enumerate(views):
                ans,unknown = fit(v[learn],y[query_ids[learn]],[v[b],test_views[j]])
                hist[where[b],j] = ans
                batch_pred.append(unknown)
            batch.append(np.column_stack(batch_pred))
        test_probs.append(np.mean(batch,axis=0))
    assert np.isfinite(hist).all()
    pred,export = decisions(hist,nine,v131,native),decisions(np.mean(test_probs,axis=0),tn,tv,tnative)
    report = dict(rows=len(rows),consensus_rows=int(consensus.sum()),consensus_errors=int((consensus&(nine!=y[rows])).sum()),
                  folds=int(len(np.unique(fold))),methods={})
    for name,h in pred.items():
        r = stats(h,nine,y[rows],g[rows],fold)
        for key,mask in [('consensus',consensus),('disputed',~consensus)]:
            r[key] = dict(fixed=int((mask&(h==y[rows])&(nine!=y[rows])).sum()),
                          harmed=int((mask&(h!=y[rows])&(nine==y[rows])).sum()))
        r['test_changes'] = int((export[name]!=tn).sum())
        report['methods'][name] = r
        print(name,r['corrected'],r['harmed'],r['fold_net'],r['consensus'],r['test_changes'],flush=True)
    for suffix in ('direct','quarter'):
        control = pred['raw_'+suffix]
        trial = pred['joint_'+suffix]
        comparison = stats(trial,control,y[rows],g[rows],fold)
        for key,mask in [('all_initial_errors',nine!=y[rows]),('consensus_errors',consensus&(nine!=y[rows]))]:
            new = mask & (trial==y[rows]) & (control!=y[rows])
            lost = mask & (trial!=y[rows]) & (control==y[rows])
            shared = mask & (trial==y[rows]) & (control==y[rows])
            comparison[key] = dict(new_fixed=int(new.sum()),lost_fixed=int(lost.sum()),both_fixed=int(shared.sum()))
            if key=='consensus_errors':
                comparison[key]['new_passengers'] = raw.PassengerId.iloc[rows[new]].tolist()
        report['joint_versus_raw_'+suffix] = comparison
    path = OUT/'predictions.npz'
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream,rows=rows,passenger=raw.PassengerId.iloc[rows].to_numpy(str),fold=fold,
                            model_names=np.array(METHODS),historical=hist,test_ids=test_ids,test_by_fold=np.asarray(test_probs))
    path.with_suffix('.tmp').replace(path)
    report['output_sha256'] = sha256(path)
    (ROOT/'reports/nov23_matched_learning_curve/mitra_nonlinear_token_trees.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    arguments=argparse.ArgumentParser()
    arguments.add_argument('--encode',action='store_true',help='Use .venv-mitra to load the frozen CPU preprocessors.')
    args=arguments.parse_args()
    with threadpool_limits(2):
        encode_inputs() if args.encode else run()
