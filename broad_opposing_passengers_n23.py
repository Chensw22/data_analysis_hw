"""Train separate outcome states for passengers in close opposite-label pairs and other passengers."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts,coordinates
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from exaone_validation_data import CATEGORICAL

REPORT=ROOT/'reports/nov23_opposing_passengers'


def match(features,g,y):
    dist,idx = cKDTree(features).query(features,k=48)
    # Nearest pairs relative to both passengers' local spacing have priority.
    scale = np.maximum(dist[:,-1],.01)
    i,j = np.nonzero((y[:,None]!=y[idx]) & (g[:,None]!=g[idx]))
    other = idx[i,j]
    cost = dist[i,j]/np.sqrt(scale[i]*scale[other])
    order = np.argsort(cost,kind='stable')
    used = np.zeros(len(y),bool)
    pos,neg = [],[]
    for ix in order:
        a,b = i[ix],other[ix]
        if used[a] or used[b]:
            continue
        if not y[a]:
            a,b = b,a
        used[[a,b]] = True
        pos.append(a)
        neg.append(b)
        if len(pos)==len(y)//5:
            break
    pos,neg = np.array(pos),np.array(neg)
    assert len(pos)>0 and np.all(y[pos]) and not np.any(y[neg]) and used.sum()==2*len(pos)
    assert not np.any(g[pos]==g[neg])
    return pos,neg


def tree(train,labels,query):
    clf = CatBoostClassifier(iterations=280,depth=5,learning_rate=.065,l2_leaf_reg=4,
                             verbose=False,random_seed=427,thread_count=2,allow_writing_files=False)
    clf.fit(train,labels,cat_features=CATEGORICAL)
    np.testing.assert_array_equal(clf.classes_,np.arange(len(np.unique(labels))))
    return clf.predict_proba(query)


def state_predictions(frame,g,y,t,q):
    x,_ = coordinates(frame,t)
    pos,neg = match(x[t],g[t],y[t])
    selected = np.zeros(len(t),bool)
    selected[np.r_[pos,neg]] = True
    rng = np.random.default_rng(527)
    random = selected.copy()
    for label in (False,True):
        ids=np.flatnonzero(y[t]==label)
        random[ids] = rng.permutation(selected[ids])
    train,query = frame.iloc[t],frame.iloc[q]
    out = {'binary':tree(train,y[t].astype(int),query)[:,1]}
    for name,flag in [('matched',selected),('shuffled',random)]:
        # 0/1 ordinary nontransported/transported, 2/3 paired nontransported/transported.
        score = tree(train,2*flag+y[t].astype(int),query)
        out[name] = score[:,1]+score[:,3]
        out[name+'_neutral'] = score[:,1]+.5*(score[:,2]+score[:,3])
        out[name+'_conditional'] = score[:,3]/(score[:,2]+score[:,3])
    return out,dict(pairs=len(pos),training=len(t))


def evaluate(label,run,report=REPORT):
    row,y,frame,g,_ = load()
    wrong,right,hard = cohorts(row,y)
    plans,history = {},{}
    directory=ART/('nov23_'+label)
    directory.mkdir(exist_ok=True)
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            fold,ref,base=z['fold'],z['ref'],z['base']
        bank,notes = {},[]
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            result,info=run(frame,g,y,t,q)
            notes.append(info)
            for k,p in result.items():
                bank.setdefault(k,np.full(len(y),np.nan))[q]=p
        hits={'reference':ref==y}
        for k,p in bank.items():
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            hits[k]=(p>=.5)==y
            hits['quarter_'+k]=np.where(ref==(base>=.5),.75*base+.25*p>=.5,ref)==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,fold_diagnostics=notes)
        if seed!='v131':
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(directory/f'{seed}.npz',row=row,y=y,fold=fold,base=base,ref=ref,**bank)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},flush=True)
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(report/(label+'.json'),dict(plans=plans,stable=stable))


if __name__=='__main__':
    with threadpool_limits(2):
        evaluate('opposing_passengers',state_predictions)
