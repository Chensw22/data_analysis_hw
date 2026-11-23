"""Test whether different uses of training labels can resolve the same strongly supported mistakes."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def main():
    row,y,frame,g,_=load()
    ids,target,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,target)
    _,releases,_=development()
    state=strata(frame)
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    hist,refs,plans={},[],{}
    wrong,right=[],[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_broad_partition_evidence/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref=z['fold'],z['base'],z['ref']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        def get(folder,keys):
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for k,v in [('row',row),('y',y),('fold',fold),('base',base),('ref',ref)]:
                    np.testing.assert_array_equal(z[k],v)
                return [z[k] for k in keys]
        native,teacher,student=get('crossfamily_unlabelled_distillation',['native','teacher','target_mix'])
        soft,rectified=get('training_error_factorial',['teacher','prob'])
        # The two directional arms reveal real labels for training false negatives and false positives, respectively.
        supplemental={'positive_expert':rectified[:,2],'negative_expert':rectified[:,4],'true_labels':rectified[:,7],
                      'soft_teacher':soft,'student':student}
        neural,smooth=get('unlabelled_neural_smoothing',['supervised','target_mix'])
        supplemental.update(neural=neural,smoothed_neural=smooth)
        bias,naive=get('training_information',['competition_corrected','competition_naive'])
        supplemental.update(corrected=bias,soft_student=naive)
        agree=(native>=.5)==(teacher>=.5)
        if seed!='v131':
            wrong.append(agree&((native>=.5)!=y))
            right.append(agree&((native>=.5)==y))
        def odds(p):return logit(p.clip(.002,.998))
        common=np.c_[odds(base),odds(native),odds(teacher)]
        extra=np.column_stack([odds(p) for p in supplemental.values()])
        votes={k:p>=.5 for k,p in supplemental.items()}
        for mode in ['parents','combined','separate_agreement']:
            x=common if mode=='parents' else np.c_[common,extra]
            v=ref.copy()
            for f in np.unique(fold):
                free=(fold==f)&(ref==(base>=.5))
                for subset in ([agree,~agree] if mode=='separate_agreement' else [np.ones(len(y),bool)]):
                    ix=np.flatnonzero(free&subset)
                    for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                        t,q=ix[a],ix[b]
                        assert not np.intersect1d(g[t],g[q]).size
                        v[q]=calibrated(x[t],x[q],y,t)>=.5
            votes[mode]=v
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        stats={}
        for k,v in votes.items():
            h=v==y
            stats[k]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()))
            if seed!='v131':hist.setdefault(k,[]).append(h)
        plans[seed]=stats
    wrong=np.logical_and.reduce(wrong)
    right=np.logical_and.reduce(right)
    assert wrong.sum()==818
    masks.update(parents_wrong=wrong,parents_wrong_and_base_wrong=wrong&masks['repeat'])
    stable={k:np.stack(hist[k]).all(0) for k in supplemental}
    bad={k:(~np.stack(hist[k])).all(0) for k in supplemental}
    counts={}
    for name,sel in [('parents_wrong',wrong),('parents_wrong_and_base_wrong',masks['parents_wrong_and_base_wrong']),
                     ('parents_right',right)]:
        panel=bad if name=='parents_right' else stable
        mat=np.column_stack(list(panel.values()))
        counts[name]=dict(total=int(sel.sum()),union=int((sel&mat.any(1)).sum()),
            individual={k:int((v&sel).sum()) for k,v in panel.items()})
    write(REPORT/'portfolio.json',dict(plans=plans,overlap=counts,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))
    print(counts,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
