"""Fit tree students on genuine training labels and soft forecasts for unlabeled passengers."""
import hashlib
import numpy as np
import torch
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_booking_canonical_n23 import load_roster
from broad_boosting_path_n23 import calibrated
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from broad_release_miss_coverage_n23 import summarize
from class_analogue_experts_n23 import BILLS
from exaone_validation_data import CATEGORICAL
from released_correction_audit_n23 import development
from tabicl_routed_context_n23 import DIGEST,WEIGHT,TabICLClassifier


def tree(frame,train,labels,weights):
    m=CatBoostClassifier(iterations=480,depth=6,learning_rate=.065,thread_count=2,random_seed=366,
                         verbose=False,allow_writing_files=False,loss_function='CrossEntropy')
    return m.fit(frame.iloc[train],labels,cat_features=CATEGORICAL,sample_weight=weights)


def predict(compact,frame,t,q,external,y,f):
    assert not np.intersect1d(t,np.r_[q,external]).size
    teacher=TabICLClassifier(n_estimators=2,model_path=WEIGHT,allow_auto_download=False,device='cpu',
                             batch_size=1,n_jobs=2,random_state=73+f,kv_cache=False,use_amp=False,use_fa3=False)
    teacher.fit(compact.iloc[t],y)
    np.testing.assert_array_equal(teacher.classes_,[False,True])
    scores=teacher.predict_proba(compact.iloc[np.r_[q,external]])[:,1]
    label_map=np.full(len(frame),np.nan)
    label_map[np.r_[q,external]]=scores
    native=tree(frame,t,y.astype(float),np.ones(len(t)))
    ordinary=native.predict_proba(frame.iloc[q])[:,1]
    seed=np.random.default_rng(837+f)
    mix=np.r_[q,seed.choice(external,len(external)-len(q),replace=False)]
    result=dict(teacher=scores[:len(q)],native=ordinary)
    for name,idx in [('external',external),('target_mix',mix),('shuffled',mix)]:
        soft=label_map[idx].copy()
        if name=='shuffled':
            # Within each native-tree risk decile, keep teacher labels and counts,
            # but reassign the teacher's finer disagreements to different people.
            prior=native.predict_proba(frame.iloc[idx])[:,1]
            codes=np.digitize(prior,np.unique(np.quantile(prior,np.arange(1,10)/10)))
            for code in np.unique(codes):
                cell=np.flatnonzero(codes==code)
                soft[cell]=seed.permutation(soft[cell])
        w=np.r_[np.ones(len(t)),np.full(len(idx),.5*len(t)/len(idx))]
        model=tree(frame,np.r_[t,idx],np.r_[y,soft],w)
        result[name]=model.predict_proba(frame.iloc[q])[:,1]
    return result


def student_frame(compact,size):
    frame=compact.copy()
    frame['BookingSize']=size
    bills=frame[BILLS]
    total=bills.sum(axis=1,min_count=1)
    frame['KnownTotal']=total
    frame['PositiveServices']=bills.gt(0).sum(axis=1)
    frame['TotalPerMember']=total/size
    for c in BILLS:frame[c+'Share']=bills[c]/(1+total)
    return frame


def main():
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    torch.set_num_threads(2)
    row,y,masks=historical()
    labels,releases,_=development()
    np.testing.assert_array_equal(labels,y)
    compact,group,size,coverage=load_roster()
    external=np.arange(len(compact)-coverage['test_rows'],len(compact))
    assert not np.intersect1d(group[external],group[row]).size
    frame=student_frame(compact,size)
    state=strata(frame.iloc[row])
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    out=ART/'nov23_crossfamily_unlabelled_distillation'
    out.mkdir(exist_ok=True)
    plans,hist,refs={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        raw={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(group[row],group[row[q]]))
            for k,p in predict(compact,frame,row[t],row[q],external,y[t],int(f)).items():
                raw.setdefault(k,np.zeros(len(y)))[q]=p
        decisions={k:v>=.5 for k,v in raw.items()}
        for k in ['native','external','target_mix','shuffled']:
            x=np.c_[logit(base.clip(.002,.998)),logit(raw['teacher'].clip(.002,.998)),logit(raw[k].clip(.002,.998))]
            vote=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=group[row[ix]]):
                    t,q=ix[a],ix[b]
                    vote[q]=calibrated(x[t],x[q],y,t)>=.5
            decisions['combined_'+k]=vote
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        stats,paired={},{}
        for k,v in decisions.items():
            h=v==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()))
            if seed!='v131':hist.setdefault(k,[]).append(h)
            compare='combined_native' if k.startswith('combined_') else 'native'
            old=decisions[compare]==y
            paired[k]=dict(fixed=int((h&~old).sum()),harmed=int((~h&old).sum()))
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,base=base,**raw)
        plans[seed]=dict(models=stats,paired=paired)
        print(seed,stats['combined_target_mix'],paired['target_mix'],flush=True)
    write(REPORT/'distilled.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):
        main()
