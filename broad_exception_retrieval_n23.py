"""Test whether outcomes contradicting the usual spending decision can be found prospectively."""
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS, load
from broad_matched_model_changes_n23 import read_bank
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_refit_dissent_n23 import features
from broad_pairwise_adjudication_n23 import paired


def main():
    row,y,frame,g,_ = load()
    risk=frame[['VRDeck','Spa','RoomService']].gt(0).any(axis=1).to_numpy()
    zero=~frame[BILLS].gt(0).any(axis=1).to_numpy()
    decision=~risk
    raw=features(frame)[0]
    exception=(decision!=y)
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(row,z['row'])
        np.testing.assert_array_equal(y,z['y'])
        historical=(z['votes']!=y[None,:,None]).all((0,2))
    plans,all_hits={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_joint_observed_state/{seed}.npz') as z:
            fold,base,ref=z['fold'],z['base'],z['ref']
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            joint=np.column_stack([z[k] for k in ['basket_conditional','basket_direct','sleep_conditional','sleep_direct']])
        pair=read_bank(row,y,fold,seed)
        all_models=np.column_stack([pair.reshape(len(y),-1),joint])
        direction=np.where(decision,-1,1)
        anchor=logit(base.clip(.003,.997))*direction
        residual=logit(all_models.clip(.003,.997))*direction[:,None]-anchor[:,None]
        # Only query-visible values define eligibility and score inputs.
        eligible=(risk|zero)&(ref==decision)&(ref==(base>=.5))
        ids=np.flatnonzero(eligible)
        x=np.c_[anchor,risk,residual]
        inputs=dict(panel=x, attributes=np.c_[x,raw])
        scores={'reference':expit(anchor)}
        for k in inputs:scores[k]=np.full(len(y),np.nan)
        score_counts={k:np.zeros(3,int) for k in scores}
        correct_counts={k:np.zeros(3,int) for k in scores}
        for f in np.unique(fold):
            pool=np.flatnonzero((fold==f)&eligible)
            for a,b in GroupKFold(3,shuffle=True,random_state=704).split(pool,groups=g[pool]):
                t,q=pool[a],pool[b]
                assert not np.intersect1d(g[t],g[q]).size
                for k,design in inputs.items():
                    if k=='panel':
                        scale=StandardScaler().fit(design[t])
                        p=LogisticRegression(C=.08,max_iter=600).fit(scale.transform(design[t]),exception[t]).predict_proba(scale.transform(design[q]))[:,1]
                    else:
                        m=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=7,min_samples_leaf=28,
                                                         learning_rate=.055,l2_regularization=8,early_stopping=False,
                                                         random_state=141)
                        p=m.fit(design[t],exception[t]).predict_proba(design[q])[:,1]
                    scores[k][q]=p
                for k,p in scores.items():
                    order=q[np.argsort(-p[q],kind='stable')]
                    for j,frac in enumerate([.01,.05,.10]):
                        top=order[:int(np.ceil(len(q)*frac))]
                        score_counts[k][j]+=len(top)
                        correct_counts[k][j]+=exception[top].sum()
        methods={}
        for k,p in scores.items():
            assert np.isfinite(p[ids]).all()
            voted=ref.copy()
            voted[ids[p[ids]>.5]]=~voted[ids[p[ids]>.5]]
            hit=voted==y
            if seed!='v131':all_hits.setdefault(k,[]).append(hit)
            methods[k]=dict(changes=paired(hit,ref==y),average_precision=float(average_precision_score(exception[ids],p[ids])),
                            selected=score_counts[k].tolist(),correct_reversals=correct_counts[k].tolist())
        plans[seed]=dict(eligible=len(ids),originally_wrong=int(exception[ids].sum()),methods=methods)
        print(seed,{k:v['changes']['fixed']-v['changes']['harmed'] for k,v in methods.items()},flush=True)
    write(ROOT/'reports/nov23_adaptive_manifest_splines/exception_retrieval.json',dict(plans=plans,
          stable_correct_on_common={k:int((np.all(v,axis=0)&historical).sum()) for k,v in all_hits.items()}))


if __name__=='__main__':
    with threadpool_limits(1):main()
