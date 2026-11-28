"""Learn the joint shape of historical dissent separately in wrong and right held-order predictions."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from dissent_likelihood_transfer_n27 import splits
from repeated_archive_misses_n26 import ART, ROOT, cohort, count


def log_density(x, center, covariance):
    delta=x-center
    sign,det=np.linalg.slogdet(covariance)
    assert sign>0
    return -.5*(np.sum(np.linalg.solve(covariance,delta.T).T*delta,axis=1)+det)


def estimate(x, strata, train, query, wrong):
    # PCA and every conditional distribution use the training passengers only.
    pca=PCA(n_components=6,whiten=True,svd_solver='full').fit(x[train])
    source=pca.transform(x[train])
    target=pca.transform(x[query])
    eye=np.eye(source.shape[1])
    out=np.empty((len(query),3))
    for cell in np.unique(strata[query]):
        selected=strata[train]==cell
        ask=strata[query]==cell
        counts=np.bincount(wrong[train][selected].astype(int),minlength=2)
        odds=np.log((counts[1]+1)/(counts[0]+1))
        densities=[]
        for label in [False,True]:
            subset=source[selected & (wrong[train]==label)]
            # Smoothing supplies a common reference distribution for sparse cells.
            prior=18.
            center=subset.sum(0)/(len(subset)+prior)
            moment=(subset.T@subset+prior*eye)/(len(subset)+prior)
            cov=moment-np.outer(center,center)
            densities.append(np.column_stack([log_density(target[ask],center,cov),
                log_density(target[ask],center,np.diag(np.diag(cov)))]))
        ratio=densities[1]-densities[0]
        out[ask]=np.column_stack([np.full(ask.sum(),odds),odds+ratio[:,1],odds+ratio[:,0]])
    return out


def run():
    frame,y,groups,f,ids,_,p,ordinary,versions=inputs()
    n=len(y)
    parent=versions['random'][:,0]
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    common=((ordinary>=.5)==parent[:,None]).all(1)
    wrong=parent[:n]!=y
    signed=np.where(parent[:,None],1-p,p)
    risk=np.median(np.where(parent[:,None],1-ordinary,ordinary),axis=1)
    # Use the class, degree of reference-model uncertainty and presence of dissent as baseline cells.
    cells=parent.astype(int)*6 + (~common)*3 + np.digitize(risk,[.18,.36])
    x=logit(np.clip(signed,.005,.995))
    x-=np.median(x,axis=1,keepdims=True)
    stored,report={},{}
    for mode,protocol in [('random','random'),('ordered','contiguous')]:
        four=(versions[mode]==parent[:,None]).all(1)
        target=four & (signed>.5).any(1)
        masks=dict(all=known, new_orders=known&fresh, seven_four=known&four[:n]&common[:n])
        parts=splits(groups,f,protocol)
        prob=np.empty((len(frame),3))
        test_votes=[]
        for fold in range(8):
            for held in [0,1,2,3,None]:
                bank=(f==fold)&known&target[:n]
                if held is None:
                    q=np.arange(n,len(frame))
                else:
                    bank &= parts!=held
                    q=np.flatnonzero((f==fold)&(parts==held))
                    assert not np.isin(groups[bank],groups[q]).any()
                z=estimate(x,cells,np.flatnonzero(bank),q,wrong)
                if held is None:test_votes.append(z)
                else:prob[q]=z
        prob[n:]=np.median(test_votes,axis=0)
        flags=(prob>0)&target[:,None]
        rules=dict(diagonal=flags[:,1],joint=flags[:,2],
                   correlation_only=flags[:,2]&~flags[:,1]&~flags[:,0],
                   common_joint=flags[:,2]&common)
        report[mode]={}
        for key,flag in rules.items():
            stored[mode+'_'+key]=flag
            report[mode][key]=dict(scopes={s:count(flag[:n],mask,wrong) for s,mask in masks.items()},
                 accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()),
                 common_test=int(np.sum(flag[n:]&common[n:])))
        print(mode,report[mode],flush=True)
    out=ART/'nov28_dissent_shape_likelihood'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz',passenger=ids,y=y,fold=f,**stored)
    (ROOT/'reports/nov23_matched_learning_curve/dissent_shape_likelihood.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
