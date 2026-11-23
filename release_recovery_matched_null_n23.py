"""Compare stable dissent on jointly wrong and jointly right passengers at matched baseline risk."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import read_opinions
from broad_partition_evidence_n23 import ART, SEEDS, write
from release_vote_information_n23 import REPORT


def matched_pairs(unanimous, ref, states, risk, groups):
    ids = np.flatnonzero(unanimous)
    # Sort by consensus confidence within an observed bill state and predicted class.
    # Pairing uses neither the outcome nor any new model scores.
    code = 3*ref.astype(int)+states
    pairs=[]
    for c in np.unique(code[ids]):
        sub=ids[code[ids]==c]
        sorted_ids=sub[np.argsort(risk[sub],kind='stable')]
        a,b=sorted_ids[:2*(len(sub)//2)].reshape(-1,2).T
        keep=(groups[a]!=groups[b]) & (abs(risk[a]-risk[b])<=.025)
        pairs.extend(zip(a[keep],b[keep]))
    return np.array(pairs).T


def test(challenge, error, a, b, component, flips):
    observed=int((challenge[a]&error[a]).sum()+(challenge[b]&error[b]).sum())
    # Expected minus observed rescued errors if each retained pair is exchanged.
    changed=(challenge[a].astype(int)-challenge[b].astype(int))*(error[b].astype(int)-error[a].astype(int))
    effects=np.bincount(component,weights=changed)
    samples=observed+flips[:,:len(effects)]@effects
    return dict(observed=observed,expected=float(observed+.5*changed.sum()),
                null95=np.quantile(samples,[.025,.975]).tolist(),
                changed_at_risk=int(np.count_nonzero(changed)),actual_challenges=int(challenge[a].sum()+challenge[b].sum()),
                upper_tail=float((1+(samples>=observed).sum())/(1+len(samples))))


def run():
    row,y,frame,g,_=load()
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        for k,v in [('row',row),('y',y)]:np.testing.assert_array_equal(z[k],v)
        votes=z['votes']
    ref=votes[0,:,0]
    eligible=(votes==ref[None,:,None]).all((0,2))
    error=ref!=y
    assert (eligible&error).sum()==983 and (eligible&~error).sum()==5343
    opposition,risks={},[]
    families={'group_counts':('multiscale_count_training','multiscale'),
              'pairs':('pair_outcome_signal','same_outcome'),
              'manifest':('joint_manifest_orders','ordered_mean'),
              'direct':('joint_manifest_orders','discriminative')}
    for seed in map(str,SEEDS):
        fold,_,opinions,_=read_opinions(seed,row,y)
        risks.append(np.where(ref, 1-opinions[:,0], opinions[:,0]))
        for k,(folder,key) in families.items():
            with np.load(ART/f'nov23_{folder}/{seed}.npz') as z:
                for n,v in [('row',row),('fold',fold),('y',y)]:np.testing.assert_array_equal(z[n],v)
                opposition.setdefault(k,[]).append((z[key]>=.5)!=ref)
    challenge={k:np.stack(v).all(0) for k,v in opposition.items()}
    challenge['any']=np.logical_or.reduce(list(challenge.values()))
    risk=np.mean(risks,0)
    a,b=matched_pairs(eligible,ref,strata(frame),risk,g)
    assert len(np.unique(np.r_[a,b]))==2*len(a)
    _,gid=np.unique(g,return_inverse=True)
    graph=coo_matrix((np.ones(len(a)),(gid[a],gid[b])),shape=(gid.max()+1,gid.max()+1))
    _,labels=connected_components(graph,directed=False)
    np.testing.assert_array_equal(labels[gid[a]],labels[gid[b]])
    _,component=np.unique(labels[gid[a]],return_inverse=True)
    rng=np.random.default_rng(162)
    # Keep all pairs connected by shared bookings together at each random exchange.
    flips=rng.integers(0,2,size=(4000,component.max()+1),dtype=np.int8)
    stats={k:test(v,error,a,b,component,flips) for k,v in challenge.items()}
    group_counts={k:dict(fixed=int((eligible&error&v).sum()),harmed=int((eligible&~error&v).sum()))
                  for k,v in challenge.items()}
    assert group_counts['any']['fixed']==178
    write(REPORT/'matched_null.json',dict(pairs=len(a),retained_errors=int(error[a].sum()+error[b].sum()),
        retained_correct=int((~error[a]).sum()+(~error[b]).sum()),components=int(component.max()+1),
        median_risk_gap=float(np.median(abs(risk[a]-risk[b]))),counts=group_counts,paired=stats))
    print(group_counts,stats['any'],flush=True)


if __name__=='__main__':
    with threadpool_limits(2):run()
