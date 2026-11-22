"""Infer class odds by sequentially reconstructing the manifest from spending."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from sklearn.metrics import log_loss
from peer_bill_correction import BILLS
from sleeper_consensus_scope import ART, ROOT
import relational_error_corrector as ref

FIELDS = ['CryoSleep','HomePlanet','Deck','Side','CabinNumber','Age','Destination','VIP']


def classifier(x,y):
    cats = list(x.select_dtypes(include=['object','str','category']).columns)
    return CatBoostClassifier(iterations=175, learning_rate=.08, depth=4, l2_leaf_reg=5,
                              cat_features=cats, one_hot_max_size=32, thread_count=2,
                              verbose=False, random_seed=2312, allow_writing_files=False).fit(x,y)


def encode(train, query, field):
    if field == 'Age':
        levels = [6,13,19,28,40,60]
    elif field == 'CabinNumber':
        levels = np.unique(train.dropna().quantile(np.arange(1,8)/8))
    else:
        return train.astype(str).to_numpy(), query.astype(str).to_numpy()
    def discretize(col):
        return np.where(col.isna(),-1,np.searchsorted(levels,col)).astype(str)
    return discretize(train),discretize(query)


def infer(frame, t, q, y):
    tx, qx = [frame.iloc[index][BILLS+['party_size']].copy() for index in (t,q)]
    prior = classifier(tx,y[t]).predict_proba(qx)[:,1]
    total = np.zeros(len(q))
    for field in FIELDS:
        lt, lq = encode(frame[field].iloc[t], frame[field].iloc[q],field)
        model = classifier(tx.assign(hypothesis=y[t].astype(int)),lt)
        idx = pd.Index(model.classes_).get_indexer(lq)
        for cls,sign in [(1,1),(0,-1)]:
            prob = model.predict_proba(qx.assign(hypothesis=cls))
            seen = idx>=0
            # Unseen manifest values are neutral, known categories receive add-one shrinkage.
            like = (prob[np.flatnonzero(seen),idx[seen]]*len(t)+1)/(len(t)+len(model.classes_))
            total[seen]+=sign*np.log(like)
        tx[field],qx[field]=lt,lq
    direct = classifier(tx,y[t]).predict_proba(qx)[:,1]
    return prior,expit(logit(prior)+total),direct


def run():
    inp,splits,fold,_,_=ref.overlap.load_authenticated()
    dev,y=inp['dev'],inp['y'].astype(bool)
    ids=pd.concat([pd.read_csv(ROOT/f'data/raw/{f}.csv',usecols=['PassengerId']).PassengerId for f in ['train','test']],ignore_index=True)
    group=ids.str.split('_').str[0]
    frame=inp['features'].assign(party_size=group.map(group.value_counts()).to_numpy()).iloc[dev].reset_index(drop=True)
    out={s:np.full(len(dev),np.nan) for s in ['prior','reverse','direct']}
    for i,(t,q) in enumerate(splits):
        for s,p in zip(out,infer(frame,t,q,y)):
            out[s][q]=p
        print('Finished fold',i,flush=True)
    with np.load(ART/'bill_likelihood_oof.npz') as z:
        np.testing.assert_array_equal(z['dev'],dev)
        forward=z['bill_posterior']
    out['bidirectional']=expit((logit(np.clip(forward,1e-8,1-1e-8))+logit(np.clip(out['reverse'],1e-8,1-1e-8)))/2)
    ref_rows=pd.read_csv(ART/'nov21_case_checks/v136_final_oof.csv')
    np.testing.assert_array_equal(ref_rows.row,dev)
    np.testing.assert_array_equal(ref_rows.fold,fold)
    np.testing.assert_array_equal(ref_rows.true,y)
    base,hard=ref_rows.v131.to_numpy(),ref_rows.hard.to_numpy()
    scores={}
    for name,p in out.items():
        assert np.isfinite(p).all()
        ok=(p>=.5)==y
        fixes, harms=ok & (base!=y),~ok & (base==y)
        scores[name]=dict(accuracy=float(ok.mean()),logloss=float(log_loss(y,p)),fixed=int(fixes.sum()),
                          harmed=int(harms.sum()),hard_fixed=int((hard&fixes).sum()),hard_harmed=int((hard&harms).sum()),
                          missed_by_forward_and_direct=int((fixes & ((forward>=.5)!=y) & ((out['direct']>=.5)!=y)).sum()),
                          fold_net=[int(fixes[fold==i].sum()-harms[fold==i].sum()) for i in range(3)])
    outdir=ART/'reverse_manifest_likelihood_n23'
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(dict(row=dev,fold=fold,y=y,p=out['reverse'],**out)).to_csv(outdir/'oof.csv',index=False)
    ref.write(ROOT/'reports/reverse_manifest_likelihood_n23/results.json',dict(rows=len(y),outcomes=scores,
              prior='Five bills and roster party size',manifest_chain=FIELDS,missing='Explicit observed missing category',
              control='Direct discrimination using same fold-specific discretized manifest, bills, and roster party size.',
              bidirectional='Equal log-odds pool with existing bill_posterior from conditional_bill_likelihood.py'))
    print(scores)


if __name__=='__main__':
    run()
