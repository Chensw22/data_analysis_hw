"""Pool dissent outcomes hierarchically across comparable passengers."""
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts
from broad_opposing_passengers_n23 import REPORT
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART,SEEDS,write


def transfer(label,codes,t,q):
    # Estimate the parent rate from siblings and a 20-observation prior.
    # Disjoint sibling sets prevent the child counts from being used twice.
    # The root prior of .5 assigns equal weight to retaining and reversing the vote.
    prior=np.full(len(q),.5)
    upper_count=np.full(len(q),len(t))
    upper_wins=np.full(len(q),label[t].sum())
    for code in codes:
        n=np.bincount(code[t],minlength=code.max()+1)[code[q]]
        win=np.bincount(code[t],weights=label[t],minlength=code.max()+1)[code[q]]
        sibling_count=upper_count-n
        sibling_wins=upper_wins-win
        assert (sibling_count>=0).all() and (sibling_wins>=0).all()
        prior=(sibling_wins+20*prior)/(sibling_count+20)
        upper_count,upper_wins=n,win
    return (upper_wins+20*prior)/(upper_count+20)


def run(data,frame,g,y):
    ref,fold,base=data['ref'],data['fold'],data['base']
    expert=data['matched_conditional']
    eligible=(ref==(base>=.5))&(ref!=(expert>=.5))
    wrong=y!=ref
    confidence=np.maximum(base,1-base)
    planet=np.unique(frame.HomePlanet,return_inverse=True)[1]
    paying=frame[BILLS].gt(0).any(axis=1).to_numpy()
    cols=(ref.astype(int),np.digitize(confidence,[.65,.8,.9]),planet,paying,
          np.digitize(np.abs(logit(expert.clip(.002,.998))),[.5,1.5]))
    def nested(order):
        values=np.zeros(len(y),int)
        levels=[]
        for k in order:
            values=values*(cols[k].max()+1)+cols[k]
            levels.append(values)
        return levels
    models={k:nested(order) for k,order in [('confidence',[0,1]),('manifest',[0,1,2,3]),
                                             ('specialist',[0,1,2,3,4])]}
    pred={k:ref.copy() for k in models}
    for f in np.unique(fold):
        ids=np.flatnonzero(fold==f)
        for a,b in GroupKFold(3,shuffle=True,random_state=719).split(ids,groups=g[ids]):
            t,q=ids[a],ids[b]
            assert not np.intersect1d(g[t],g[q]).size
            t,q=t[eligible[t]],q[eligible[q]]
            for k,codes in models.items():
                pred[k][q]^=transfer(wrong,codes,t,q)>.5
    return pred


def main():
    row,y,frame,g,_=load()
    wrong,right,hard=cohorts(row,y)
    plans,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        with np.load(ART/f'nov23_opposing_passengers/{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            np.testing.assert_array_equal(y,z['y'])
            ref=z['ref']
            pred=run(z,frame,g,y)
        pred['reference']=ref
        hit={k:v==y for k,v in pred.items()}
        plans[seed]={k:paired(v,ref==y) for k,v in hit.items()}
        if seed!='v131':
            for k,v in hit.items():
                history.setdefault(k,[]).append(v)
        print(seed,{k:v['fixed']-v['harmed'] for k,v in plans[seed].items()})
    baseline=np.stack(history['reference'])
    stable={}
    for k,h in history.items():
        h=np.stack(h)
        stable[k]=dict(hard_stable=int((hard&h.all(0)).sum()),hard_ever=int((hard&h.any(0)).sum()),
                       common_stable=int((wrong&h.all(0)).sum()),stable_harms=int((right&~h.any(0)).sum()),
                       gain95_pp=interval((h.astype(float)-baseline).mean(0),g))
    write(REPORT/'dispute_cells.json',dict(plans=plans,stable=stable))


if __name__=='__main__':
    main()
