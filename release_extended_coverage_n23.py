"""Test whether new outcome objectives break joint mistakes of the current release slate."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_booking_policy_replay_n23 import interval
from broad_pairwise_adjudication_n23 import paired, read_opinions
from broad_partition_evidence_n23 import ART, SEEDS, write
from release_vote_information_n23 import REPORT, bookend


def odds(p):
    return logit(p.clip(.003,.997))


def train_offset(base, features, y, t, q):
    delta=features-base[:,None]
    train=np.c_[np.ones(len(t)), delta[t]]
    test=np.c_[np.ones(len(q)), delta[q]]
    def loss(w):
        z=base[t]+train@w
        return (np.sum(np.logaddexp(0,z)-y[t]*z)+20*w@w,
                train.T@(expit(z)-y[t])+40*w)
    fit=minimize(loss,np.zeros(train.shape[1]),method='L-BFGS-B',jac=True)
    if not fit.success:
        raise RuntimeError(fit.message)
    return base[q]+test@fit.x>=0


def main():
    row,y,_,g,_=load()
    families={'multiscale':('multiscale_count_training','multiscale'),
              'paired':('pair_outcome_signal','same_outcome'),
              'generative':('joint_manifest_orders','ordered_mean'),
              'discriminative':('joint_manifest_orders','discriminative')}
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        released=z['votes']
    all_five_wrong=np.all(released!=y[None,:,None],axis=(0,2))
    all_five_right=np.all(released==y[None,:,None],axis=(0,2))
    metrics,stable,ref_hits,raw_hits={},{},[],[]
    for seed in [*map(str,SEEDS),'v131']:
        fold,ref,opinions,_=read_opinions(seed,row,y)
        base=opinions[:,0]
        additions=[]
        for folder,key in families.values():
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for k,v in [('row',row),('fold',fold),('y',y),('ref',ref)]:
                    np.testing.assert_array_equal(z[k],v)
                additions.append(z[key])
        extras=np.column_stack(additions)
        with np.load(ART/('nov23_budget_quarter_replay/original.npz' if seed=='v131'
                           else f'nov23_simplex_likelihood/{seed}.npz')) as z:
            np.testing.assert_array_equal(z['fold'],fold)
            budget=z['alternative' if seed=='v131' else 'budget_prior']
        with np.load(ART/f'nov23_pretrained_spending_views/{seed}.npz') as z:
            np.testing.assert_array_equal(z['fold'],fold)
            foundation=z['budget']
        simple=odds(np.c_[budget,foundation])
        extended=np.c_[simple,odds(extras)]
        cuts=(opinions>=.5).all(1) | (opinions<.5).all(1)
        free=(base>=.5)==ref
        methods={}
        for key,x in [('two_budgets',simple),('all_scores',extended),('separate_agreement',extended)]:
            vote=ref.copy()
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&free)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ids,groups=g[ids]):
                    assert not np.intersect1d(g[ids[a]],g[ids[b]]).size
                    for mask in ([cuts,~cuts] if key=='separate_agreement' else [np.ones(len(y),bool)]):
                        t,q=ids[a][mask[ids[a]]],ids[b][mask[ids[b]]]
                        if len(t) and len(q):
                            vote[q]=train_offset(odds(base),x,y,t,q)
            methods[key]=vote
        h={k:v==y for k,v in methods.items()}
        metrics[seed]={k:paired(v,ref==y) for k,v in h.items()}
        if seed!='v131':
            for k,v in h.items():
                stable.setdefault(k,[]).append(v)
            raw_hits.append((extras>=.5)==y[:,None])
            ref_hits.append(ref==y)
        else:
            same=(extras>=.5)==y[:,None]
            archive={m:{'rows':int(mask.sum()),
                        'by_model':dict(zip(families,same[mask].sum(0).tolist())),
                        'any_right':int(same[mask].any(1).sum()),
                        'right_by_combiner':{k:int(v[mask].sum()) for k,v in h.items()}}
                     for m,mask in [('joint_wrong',all_five_wrong),('joint_right',all_five_right)]}
            limits=[bookend(np.c_[v,(extras>=.5)],y) for v in released]
    all_correct=np.stack(raw_hits).all(0)
    simultaneous=np.stack(raw_hits).any(2).all(0)
    groups={'all':np.ones(len(y),bool),'joint_wrong':all_five_wrong,'joint_right':all_five_right,
            'three_grouped_baseline_misses':~np.stack(ref_hits).any(0)}
    overlap={k:dict(rows=int(mask.sum()),each=dict(zip(families,all_correct[mask].sum(0).tolist())),
                    any_stable=int(all_correct[mask].any(1).sum()),at_least_one_each_split=int(simultaneous[mask].sum()))
             for k,mask in groups.items()}
    numbers={k:dict(stable={m:int((np.stack(v).all(0)&mask).sum()) for m,mask in groups.items()},
                    mean_gain95_pp=interval((np.stack(v).astype(float)-np.stack(ref_hits)).mean(0),g))
             for k,v in stable.items()}
    write(REPORT/'extended.json', dict(metrics=metrics, original_replay=archive, retrospect_with_extra_votes=limits,
                                       three_grouped_replays=overlap, combined=numbers))
    print(archive,overlap['joint_wrong'],metrics,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
