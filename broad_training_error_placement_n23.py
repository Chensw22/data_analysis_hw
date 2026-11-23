"""Compare the placement of directional training corrections at equal decision counts."""
import numpy as np
from scipy.special import logit, expit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART,SEEDS,historical,write
from broad_prediction_assisted_training_n23 import REPORT
from broad_release_miss_coverage_n23 import summarize
from released_correction_audit_n23 import development


def rank_transfer(source,destination,cells,seed):
    # Give each recipient the donor value with the same within-cell rank; no query labels.
    rng=np.random.default_rng(seed)
    output=np.empty_like(source)
    for cell in np.unique(cells):
        ix=rng.permutation(np.flatnonzero(cells==cell))
        ranked=ix[np.argsort(source[ix],kind='stable')]
        output[ranked]=np.sort(destination[ix])
        np.testing.assert_array_equal(np.sort(output[ix]),np.sort(destination[ix]))
    return output


def main():
    row,y,frame,g,_=load()
    ids,labels,masks=historical()
    np.testing.assert_array_equal(row,ids)
    np.testing.assert_array_equal(y,labels)
    _,releases,_=development()
    state=strata(frame)
    masks={**masks,'paid_transported':masks['repeat']&(state==2)&y,'zero_stayed':masks['repeat']&(state==0)&~y}
    hist,plans,refs={},{},[]
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_training_error_factorial/{seed}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['y'],y)
            fold,base,ref,prob,teacher=z['fold'],z['base'],z['ref'],z['prob'],z['teacher']
        np.testing.assert_array_equal(ref,releases[131] if seed=='v131' else base>=.5)
        # Match all eight predictions to the distribution of the all-true-label fit.
        # Score ties near the threshold can be resolved independently of the outcome.
        bands=np.zeros(len(y),int)
        for f in np.unique(fold):
            ix=fold==f
            bands[ix]=np.digitize(teacher[ix],np.quantile(teacher[ix],np.arange(1,6)/6))
        cells=(fold*3+state)*6+bands
        normal=np.column_stack([rank_transfer(p,prob[:,7],cells,251) for p in prob.T])
        np.testing.assert_allclose(normal[:,7],prob[:,7],atol=0)
        vote={'full':prob[:,7]>=.5}
        for i,k in [(2,'positive'),(4,'negative')]:
            vote[k+'_native']=prob[:,i]>=.5
            vote[k+'_threshold_only']=rank_transfer(prob[:,7],prob[:,i],cells,251)>=.5
            vote[k+'_equal_count']=normal[:,i]>=.5
            for c in np.unique(cells):
                mask=cells==c
                assert vote[k+'_threshold_only'][mask].sum()==vote[k+'_native'][mask].sum()
        def odds(p):
            return logit(p.clip(.002,.998))
        core=np.c_[odds(base),odds(teacher),odds(prob[:,7])]
        vote['equal_average']=normal.mean(axis=1)>=.5
        for mode,views in [('base',None),('native',prob),('equal',normal)]:
            x=core if views is None else np.c_[core,odds(views[:,:7])-odds(prob[:,7,None])]
            v=ref.copy()
            for f in np.unique(fold):
                ix=np.flatnonzero((fold==f)&(ref==(base>=.5)))
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=g[ix]):
                    t,q=ix[a],ix[b]
                    v[q]=calibrated(x[t],x[q],y,t)>=.5
            vote['combined_'+mode]=v
        anchor=ref==y
        if seed!='v131':refs.append(anchor)
        result={}
        for mode,v in vote.items():
            h=v==y
            result[mode]=dict(accuracy=float(h.mean()),fixed=int((h&~anchor).sum()),harmed=int((~h&anchor).sum()))
            if seed!='v131':hist.setdefault(mode,[]).append(h)
        plans[seed]=result
        print(seed,result['combined_equal'],flush=True)
    write(REPORT/'placement.json',dict(plans=plans,grouped={
        k:summarize(np.stack(h),np.stack(refs),masks) for k,h in hist.items()}))


if __name__=='__main__':
    with threadpool_limits(2):main()
