"""Learn which generative dissent to accept and test on bookings absent from the historical pool."""
import json
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from common_exception_audit_n25 import inputs
from disjoint_tree_fields_n25 import ART,DEST
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_teacher_n25 import OUT as REF

OUT=ART/'nov25_passenger_joint_likelihood'


def collect(rows,fold,y,g,prefix):
    target=None
    for f in np.unique(fold):
        mask=fold==f
        q=rows[mask]
        t=rows[(~mask)&~np.isin(g[rows],g[q])]
        with np.load(OUT/f'{prefix}{f}.npz') as z:
            np.testing.assert_array_equal(z['t'],t)
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['train_y'],y[t])
            names=['outcome_only','direct','chain_0','chain_1',*[n for n in z.files if n.startswith(('omit_','restore_'))]]
            if target is None:
                fields=names
                target=np.empty((len(rows),len(fields)))
            assert names==fields
            target[mask]=np.column_stack([z[n] for n in fields])
    return target


def design(archive,learned,raw_features):
    votes=(archive>=.5).sum(1)
    ref=votes>34
    error_direction=np.where(ref,-1.,1.)[:,None]
    # Orient every probability toward reversing the majority label.
    alt=error_direction*logit(learned.clip(.003,.997))
    pop=error_direction*logit(archive.clip(.003,.997))
    ar=np.column_stack([ref,np.minimum(votes,68-votes)/34,np.quantile(pop,[0,.1,.25,.5,.75,.9,1],axis=1).T])
    features={'archive':ar,'archive_and_sources':np.column_stack([ar,alt]),
              'with_passenger':np.column_stack([ar,alt,raw_features])}
    return features,ref,np.minimum(votes,68-votes)


def counts(ref,truth,select,scope):
    return {'rows':int(scope.sum()),'errors':int((scope&(ref!=truth)).sum()),
            'fixed':int((scope&select&(ref!=truth)).sum()),'harmed':int((scope&select&(ref==truth)).sum())}


def run():
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    a,b=inputs(raw)
    rawx=np.column_stack([a,b])
    full=collect(np.arange(len(y)),f,y,g,'fold')
    old=collect(rows,hf,y,g,'historical')
    total=len(y)
    combined=np.r_[np.arange(total),rows]
    x,ref,minor=design(np.vstack([np.column_stack(v) for v in panels]),np.vstack([full,old]),rawx[combined])
    truth=y[combined]
    error=truth!=ref
    # Any alternate crossing exactly 0.5 must follow the same binary tie convention as the archive.
    votes=np.vstack([full,old])>=.5
    eligible=(votes!=ref[:,None]).any(1)
    known=np.full(len(y),-1)
    for k in np.unique(f):
        with np.load(REF/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],np.flatnonzero(f==k))
            known[z['q']]=z['nine']
    fresh=~np.isin(g,g[rows])
    protocols={'same_fitted_base':[],'historical_to_new_orders':[(total+np.arange(len(rows)),np.flatnonzero(fresh))]}
    for k in np.unique(f):
        outer=np.flatnonzero(f==k)
        for a,b in GroupKFold(4,shuffle=True,random_state=213).split(outer,groups=g[outer]):
            protocols['same_fitted_base'].append((outer[a],outer[b]))
    result={}
    for protocol,splits in protocols.items():
        output={}
        for method in ['archive','archive_and_sources','with_passenger','complete_agreement_only','boosted']:
            probs=[];ids=[]
            for t,q in splits:
                assert not np.isin(g[combined[q]],g[combined[t]]).any()
                t=t[eligible[t]]
                focus=method=='complete_agreement_only'
                if focus:t=t[minor[t]==0]
                active=eligible[q]&((minor[q]==0) if focus else True)
                pred=np.zeros(len(q))
                if active.any():
                    matrix=x[method if method in x else 'with_passenger']
                    if method=='boosted':
                        model=CatBoostClassifier(iterations=160,depth=4,learning_rate=.04,l2_leaf_reg=6,
                                thread_count=2,verbose=False,allow_writing_files=False,random_seed=175)
                    else:
                        model=LogisticRegression(C=.09,max_iter=400,solver='liblinear')
                    if np.unique(error[t]).size==2:
                        model.fit(matrix[t],error[t])
                        pred[active]=model.predict_proba(matrix[q[active]])[:,1]
                    else:
                        pred[active]=(error[t].sum()+1)/(len(t)+2)
                probs.append(pred);ids.append(q)
            ix=np.concatenate(ids)
            p=np.concatenate(probs)
            assert len(np.unique(ix))==len(ix) and max(ix)<total
            accept=p>=.5
            matched=(known[ix]>=0)&(known[ix]==ref[ix])
            groups={'all':np.ones(len(ix),bool),'near':minor[ix]<=3,'unanimous':minor[ix]==0,
                    'unanimous_predicted_true':(minor[ix]==0)&ref[ix],'unanimous_predicted_false':(minor[ix]==0)&~ref[ix]}
            summary={k:counts(ref[ix],truth[ix],accept,m) for k,m in groups.items()}
            summary['vs_known_nine']=counts(known[ix],truth[ix],accept&matched,known[ix]>=0)
            summary['by_fold_net_nine']=[int(((truth[ix]!=known[ix])&accept&matched&(f[ix]==k)).sum()-
                                              ((truth[ix]==known[ix])&accept&matched&(f[ix]==k)).sum()) for k in np.unique(f[ix])]
            output[method]=summary
            print(protocol,method,[(k,summary[k]['fixed'],summary[k]['harmed']) for k in ('all','unanimous','vs_known_nine')])
        result[protocol]=output
    DEST.with_name('screen_joint_disagreements.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
