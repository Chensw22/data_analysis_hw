"""Compare native missing values and stochastic chained numeric imputation in a three-tree ensemble."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import REPORT
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import BILLS
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development

NUM = ['Age','CabinNumber',*BILLS]
WEIGHTS = np.array([.4766,.2901,.2333])


def feature_table(frame, numeric):
    amounts = np.expm1(numeric[:,2:])
    total = np.nansum(amounts,axis=1)
    total[np.isnan(amounts).all(1)] = np.nan
    # Censoring stays visible to both alternatives, including after imputation.
    flags = frame[NUM].isna().to_numpy(float)
    return np.column_stack([numeric, flags, np.log1p(total), (amounts>0).sum(1),
                            amounts/(1+total[:,None]), np.log1p(total)/np.maximum(1,numeric[:,0]),
                            np.log1p(total)/frame.BookingSize.to_numpy(), np.log(frame.BookingSize)])


def ensemble(frame, numeric, cat, t, q, seed, labels):
    derived = feature_table(frame,numeric)
    encoded = np.column_stack([cat,derived])
    tab = pd.concat([frame[CATEGORICAL],pd.DataFrame(derived,index=frame.index).add_prefix('v')],axis=1)
    models = [CatBoostClassifier(iterations=420, depth=6, learning_rate=.055, random_seed=seed,
                                 thread_count=2, allow_writing_files=False,verbose=False),
              LGBMClassifier(n_estimators=320,num_leaves=17,min_child_samples=30,learning_rate=.045,
                             verbosity=-1,n_jobs=2,random_state=seed),
              XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.055,min_child_weight=5,
                            n_jobs=2,tree_method='hist',random_state=seed)]
    result = []
    for i,model in enumerate(models):
        train, query = (tab.iloc[t],tab.iloc[q]) if i==0 else (encoded[t],encoded[q])
        model.fit(train,labels,**({'cat_features':CATEGORICAL} if i==0 else {}))
        result.append(model.predict_proba(query)[:,1])
    return np.column_stack(result) @ WEIGHTS


def infer(frame, y, t, q, seed):
    cat = OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(frame.iloc[t][CATEGORICAL]).transform(frame[CATEGORICAL])
    numeric = frame[NUM].to_numpy(float).copy()
    numeric[:,2:] = np.log1p(numeric[:,2:])
    arr = np.column_stack([numeric,cat,np.log(frame.BookingSize)])
    idx = np.r_[t,q]
    native, chained = [], []
    for k in range(2):
        imp = IterativeImputer(max_iter=5, sample_posterior=True, skip_complete=True, min_value=0,
                               random_state=307+seed+37*k)
        imp.fit(arr[t])
        changed = numeric.copy()
        new = imp.transform(arr[idx])[:,:len(NUM)]
        # Truncate only imputed values to a range actually observed in the training fold.
        known = np.isfinite(numeric[idx])
        np.testing.assert_allclose(new[known],numeric[idx][known])
        changed[idx] = np.where(known,numeric[idx],np.minimum(new,np.nanmax(numeric[t],axis=0)))
        native.append(ensemble(frame,numeric,cat,t,q,133+k,y[t]))
        chained.append(ensemble(frame,changed,cat,t,q,133+k,y[t]))
    return {'native':np.mean(native,axis=0), 'chained':np.mean(chained,axis=0)}


def main():
    row,y,frame,group,_ = load()
    ids, labels,masks = historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    _, releases,_ = development()
    state = strata(frame)
    missing = frame[NUM].isna().any(axis=1).to_numpy()
    masks = {**masks,'paid_transported':masks['repeat']&(state==2)&y,
             'zero_stayed':masks['repeat']&(state==0)&~y,'numeric_complete_repeat':masks['repeat']&~missing}
    out = ART/'nov23_chained_tree_ensemble'
    out.mkdir(exist_ok=True)
    history,refs,plans = {},[],{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(z[k],v)
            fold,base,ref = z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        raw = {k:np.zeros(len(y)) for k in ('native','chained')}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f) & ~np.isin(group,group[q]))
            for k,v in infer(frame,y,t,q,int(f)).items():
                raw[k][q]=v
        votes={k:p>=.5 for k,p in raw.items()}
        for k in raw:
            x = np.c_[logit(base.clip(.002,.998)),logit(raw[k].clip(.002,.998))]
            vote=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=group[ix]):
                    t,q=ix[a],ix[b]
                    vote[q]=calibrated(x[t],x[q],y,t)>=.5
            votes['combined_'+k]=vote
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        stats={}
        for k,v in votes.items():
            hit=v==y
            stats[k]=dict(accuracy=float(hit.mean()),fixed=int((hit&~anchor).sum()),harmed=int((~hit&anchor).sum()))
            if seed!='v131': history.setdefault(k,[]).append(hit)
        paired = {}
        for k,m in [('numeric_complete',~missing),('numeric_incomplete',missing)]:
            delta=(votes['chained']==y).astype(int)-(votes['native']==y)
            paired[k]=dict(rows=int(m.sum()),fixed=int(((delta>0)&m).sum()),harmed=int(((delta<0)&m).sum()))
        plans[seed]=dict(models=stats,matched=paired)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**raw)
        print(seed,stats,flush=True)
    write(REPORT/'chained.json',dict(plans=plans,counts={k:int(m.sum()) for k,m in masks.items()},
           grouped={k:summarize(np.stack(h),np.stack(refs),masks) for k,h in history.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
