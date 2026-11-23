"""Fit the ten original leaf experts for the official roster, keeping full model budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/nov23_specialist_full_manifest'
SEED = 20261112


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data():
    a = pd.read_csv(ROOT/'data/raw/train.csv')
    b = pd.read_csv(ROOT/'data/raw/test.csv')
    # Only training targets are read. Test positions are never used to index targets.
    y = a.pop('Transported').to_numpy(bool)
    raw = pd.concat([a, b], ignore_index=True).drop(columns='Name')
    group = raw.PassengerId.str.split('_').str[0].to_numpy()
    size = pd.Series(group).map(pd.Series(group).value_counts()).to_numpy()
    t = np.arange(len(a))
    q = np.arange(len(a),len(raw))
    assert len(t)==8693 and len(q)==4277 and not np.intersect1d(group[t],group[q]).size
    return raw, group, size, y, t, q


def transformed(raw, mapping):
    df = raw.drop(columns='PassengerId').copy()
    berth = df.pop('Cabin').str.split('/', expand=True)
    df['Deck'], df['CabinNumber'], df['Side'] = berth[0], pd.to_numeric(berth[1]), berth[2]
    for key in ['CryoSleep','VIP']:
        df[key] = df[key].map(mapping)
    return df


def orion(raw, y, t, q):
    import sys
    import torch
    from orion_bix_probe import CHECKPOINT, MANIFEST_SHA, VENDOR, WEIGHT_SHA, sha
    assert sha(VENDOR/'SOURCE.json') == MANIFEST_SHA and sha(CHECKPOINT) == WEIGHT_SHA
    for name, checksum in json.loads((VENDOR/'SOURCE.json').read_text())['vendor_files_sha256'].items():
        assert sha(VENDOR/name) == checksum
    sys.path.insert(0, str(VENDOR/'src'))
    from orion_bix.sklearn.classifier import OrionBixClassifier
    torch.set_num_threads(3)
    frame = transformed(raw, {True:'yes', False:'no'})
    support = np.sort(np.random.default_rng(SEED).choice(t,1536,replace=False))
    model = OrionBixClassifier(n_estimators=2,batch_size=1,use_amp=False,device='cpu',n_jobs=3,
        model_path=str(CHECKPOINT),allow_auto_download=False,checkpoint_version=CHECKPOINT.name,random_state=121)
    model.fit(frame.iloc[support], y[support].astype(int))
    assert list(model.classes_) == [0,1] and model.model_.row_attention_type == 'standard'
    p = np.concatenate([model.predict_proba(frame.iloc[batch])[:,1] for batch in np.array_split(q,6)])
    return p, dict(weight_sha256=WEIGHT_SHA, support=support)


def carte(raw, y, t, q):
    import torch
    from carte_passengers_n23 import OUT as assets, REPORT, ObservedClassifier, Table2GraphTransformer
    w = assets/'packages/carte_ai/data/etc/kg_pretrained.pt'
    vector = assets/'manifest_words.bin'
    assert digest(w) == json.loads((REPORT/'weights.json').read_text())['weights_sha256']
    assert digest(vector) == json.loads((REPORT/'words.json').read_text())['compact_sha256']
    df = transformed(raw,{True:'true',False:'false'})
    torch.set_num_threads(3)
    prep = Table2GraphTransformer(fasttext_model_path=str(vector))
    xt = prep.fit_transform(df.iloc[t],y[t].astype(np.float32))
    xq = prep.transform(df.iloc[q])
    clf = ObservedClassifier(load_pretrain=True,num_model=1,max_epoch=14,early_stopping_patience=3,
                             batch_size=96,n_jobs=1,device='cpu',random_state=804,disable_pbar=True)
    clf.loss_history, clf.capture_prefix = [], False
    clf.fit(xt,y[t].astype(int))
    p = np.concatenate([clf.predict_proba(xq[i:i+96]) for i in range(0,len(q),96)])
    return p, dict(weight_sha256=digest(w),vectors_sha256=digest(vector),epochs=len(clf.loss_history))


def fit_member(key, raw, group, size, y, t, q):
    if key == 'orion':
        return orion(raw,y,t,q)
    if key == 'native_carte':
        return carte(raw,y,t,q)
    from exaone_validation_data import CATEGORICAL, compact_features
    x = compact_features(raw).reset_index(drop=True)
    if key == 'projection_shapes':
        from projection_shapes_n23 import BILLS, matrices, fit
        bill=x[BILLS].fillna(0)
        total=bill.sum(axis=1).to_numpy()
        features=np.c_[matrices(x)[0],x.CabinNumber.fillna(900)/1000,x.CabinNumber.isna(),
                       np.log(size),np.log1p(total)/6,bill.div(1+total,axis=0)]
        p=fit(features,y,[np.ones(len(raw),bool),total>0,total==0],t,q)['projection_shapes']
    elif key == 'manifest_likelihood':
        from manifest_likelihood_tree_n23 import encode, BLOCKS, ManifestTree
        values, widths=encode(x,size)
        vals=[]
        for label in [False,True]:
            selected=t[y[t]==label]
            learner=ManifestTree(BLOCKS,widths,3,2301+int(label))
            learner.learn(values,selected)
            vals.append(learner.infer(values,q)[0]+np.log(len(selected)/len(t)))
        p=expit(vals[1]-vals[0])
    elif key == 'class_analogues':
        from class_analogue_experts_n23 import design, soft_experts
        xi, phi=design(x,t)
        p=soft_experts(xi,phi,y,t,q,True)[1]
    elif key == 'variational_gp':
        import torch
        from passenger_variational_gp_n23 import design, fit
        torch.set_num_threads(2)
        _,phi=design(x,t)
        p=fit(np.c_[phi,np.log1p(size)],y,t,q)[0]
    elif key == 'field_factors':
        import torch
        from grouped_field_factors_n23 import encode, fit
        torch.set_num_threads(2)
        p=fit(encode(raw),torch.from_numpy(y.astype(np.float32)),t,q,3)
    elif key == 'registration_only':
        from conditional_view_density_n23 import fit_score
        register=x[CATEGORICAL].copy()
        for col in CATEGORICAL:
            register[col]=pd.Categorical(register[col]).codes
        register['AgeBand']=np.where(x.Age.isna(),7,np.digitize(x.Age,[6,13,18,25,40,60]))
        values=register.to_numpy(float)
        p=expit(fit_score(values[t],y[t],values[q],np.ones(values.shape[1],bool)))
    elif key == 'budget_prior':
        from spend_simplex_likelihood_n23 import design,learn
        p=learn(design(x)[0],y,t,q)
    elif key == 'bayesian_forest':
        from importlib.metadata import version
        from bayesian_manifest_forest_n23 import design, predict
        assert version('stochtree')=='0.4.5'
        p=predict(design(x,t,size),y,t,q,SEED,360,384)['bart']
    else:
        raise ValueError(key)
    return p, {}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('member')
    parser.add_argument('--check-archive',action='store_true')
    args=parser.parse_args()
    raw,group,size,y,t,q=data()
    name=args.member
    OUT.mkdir(exist_ok=True)
    start=time.monotonic()
    if args.check_archive:
        with np.load(ROOT/f'artifacts/nov23_group_kernel/{SEED}.npz') as z:
            row, fold = z['row'], z['fold']
            np.testing.assert_array_equal(y[row], z['y'])
        t, q = row[fold!=0],row[fold==0]
        assert not np.intersect1d(group[t],group[q]).size
    pred,info=fit_member(name,raw,group,size,y,t,q)
    pred=np.asarray(pred)
    assert pred.shape==(len(q),) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    if args.check_archive:
        # Keep the gold source map identical to the original ten-expert study.
        definitions = {'projection_shapes': ('projection_shapes','projection_shapes'),
            'orion': ('pretrained_recheck','orion'), 'variational_gp': ('variational_gp','p'),
            'native_carte': ('carte','p'), 'manifest_likelihood': ('manifest_tree','manifest_tree'),
            'class_analogues': ('class_analogues','class_linear'), 'field_factors': ('field_factors','order3'),
            'registration_only': ('bill_matched_register','register'), 'budget_prior': ('simplex_likelihood','budget_prior'),
            'bayesian_forest': ('bart_long','bart')}
        folder,col=definitions[name]
        with np.load(ROOT/f'artifacts/nov23_{folder}/{SEED}.npz') as z:
            np.testing.assert_array_equal(z['row'][fold==0], q)
            archived = z[col][fold==0]
        discrepancy = dict(rows=len(q),max_probability_difference=float(np.max(np.abs(pred-archived))),
                           mean_probability_difference=float(np.mean(np.abs(pred-archived))),
                           changed_votes=int(((pred>=.5)!=(archived>=.5)).sum()))
        (OUT/f'parity_{name}.json').write_text(json.dumps(discrepancy,indent=2)+'\n')
        print(name, discrepancy, flush=True)
        return
    np.savez_compressed(OUT/f'{name}.npz',prediction=pred,passenger=raw.PassengerId.iloc[q].to_numpy(str),
                        seconds=time.monotonic()-start,**info)
    print(name,len(q),round(time.monotonic()-start,1),flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
