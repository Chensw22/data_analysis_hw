"""Learn whether actual companions are more recognizable for each passenger outcome."""
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_matched_companions_n23 import data, donor_ids
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from class_analogue_experts_n23 import design

REPORT = ROOT/'reports/nov23_booking_link_authenticity'


def tree(x, y):
    return CatBoostClassifier(iterations=220, depth=5, learning_rate=.065, l2_leaf_reg=5,
                              thread_count=2, verbose=False, allow_writing_files=False,
                              random_seed=391).fit(x, y)


def fold_scores(frame, groups, size, peer, y, train, query, seed):
    own, raw = design(frame, train)
    raw = np.c_[raw, np.log(size)]
    rng = np.random.default_rng(410+seed)
    fake = donor_ids(own, groups, size, train, train, rng)[:,0]
    fake_query = donor_ids(own, groups, size, train, query, rng)[:,0]
    def pair(focal, companion):
        return np.c_[raw[focal],peer[companion]]
    correct, swapped = pair(train,train),pair(train,fake)
    q, shuffled_q = pair(query,query),pair(query,fake_query)
    ordinary = tree(correct,y[train]).predict_proba(q)[:,1]
    X = np.vstack([correct,swapped])
    outcome = np.tile(y[train].astype(int),2)
    genuine = np.repeat([1,0],len(train))
    keys, inv = np.unique(groups[train],return_inverse=True)
    flip = rng.integers(0,2,len(keys))[inv]
    result = {'direct':ordinary}
    auc = {}
    for name, tag in [('link',genuine),('placebo',genuine^np.tile(flip,2))]:
        model = tree(X,2*outcome+tag)
        np.testing.assert_array_equal(model.classes_,np.arange(4))
        p = model.predict_proba(np.vstack([q,shuffled_q])).reshape(2,len(q),2,2)
        qjoint = p[0]
        # The four classes encode (outcome, actual peer). Marginalize only the desired axis.
        true_p = qjoint[:,1,1]/qjoint[:,:,1].sum(1)
        fake_p = qjoint[:,1,0]/qjoint[:,:,0].sum(1)
        result[name]=true_p
        result[name+'_shift']=logit(true_p.clip(.002,.998))-logit(fake_p.clip(.002,.998))
        label = np.repeat([1,0],len(q))
        auc[name]=float(roc_auc_score(label,p[:,:,:,1].sum(2).ravel()))
    return result,auc


def main():
    row,y,_,groups,_=load()
    wrong,right,hard=cohorts(row,y)
    frame,g,size,peer,_=data(row)
    np.testing.assert_array_equal(groups,g.astype(int))
    many=size>1
    output=ART/'nov23_booking_link_authenticity'
    output.mkdir(exist_ok=True)
    plans,hist={},{}
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,op,_=read_opinions(seed,row,y)
        base=op[:,0]
        anchor=logit(base.clip(.002,.998))
        scores,aucs={},[]
        for f in np.unique(fold):
            q=np.flatnonzero((fold==f)&many)
            t=np.flatnonzero((fold!=f)&many&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            values,auc=fold_scores(frame,g,size,peer,y,t,q,int(f))
            aucs.append(auc)
            for name,p in values.items():
                initial=np.zeros(len(y)) if name.endswith('shift') else base.copy()
                scores.setdefault(name,initial)[q]=p
        keep=many & (ref==(base>=.5))
        hits={'reference':ref==y}
        for name,p in scores.items():
            assert np.isfinite(p).all()
            if name.endswith('shift'):
                for weight in [1,.25]:
                    hits[f'{name}_{weight}']=np.where(keep,anchor+weight*p>=0,ref)==y
            else:
                hits[name]=np.where(many,p>=.5,ref)==y
                hits['quarter_'+name]=np.where(keep,.75*base+.25*p>=.5,ref)==y
        metrics={k:paired(v,ref==y) for k,v in hits.items()}
        plans[seed]=dict(changes=metrics,authenticity_auc=aucs)
        if seed!='v131':
            for k,v in hits.items():hist.setdefault(k,[]).append(v)
        np.savez_compressed(output/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**scores)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in metrics.items()},aucs,flush=True)
    r=np.stack(hist['reference'])
    summary={k:dict(common_correct=int((np.all(v,axis=0)&wrong).sum()),hard_stable=int((np.all(v,axis=0)&hard).sum()),
                    hard_ever=int((np.any(v,axis=0)&hard).sum()),stable_harms=int((~np.any(v,axis=0)&right).sum()),
                    gain95_pp=interval((np.stack(v).astype(float)-r).mean(0),groups)) for k,v in hist.items()}
    write(REPORT/'links.json',dict(plans=plans,stable=summary,multi=int(many.sum()),multi_hard=int((many&hard).sum())))


if __name__=='__main__':
    with threadpool_limits(2):main()
