"""Exchange entire booking error vectors while keeping observed expert vote portfolios fixed."""
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from audit_completion_coverage_n23 import PAST,LATEST
from audit_service_distance_rescue_n23 import cached
from broad_bill_neighbor_decision_n23 import opinions
from broad_geographic_transfer_n23 import BILLS, load
from broad_matched_model_changes_n23 import read_bank
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from broad_unanimous_dissent_value_n23 import SOURCES
from broad_tree_field_dividends_n23 import hard_cohorts

REPORT=ROOT/'reports/nov23_stable_rescue_placebo'


def family_scores(seed,row,y,frame):
    fold,base,ref,bills,near=opinions(seed,row,y,frame.BookingSize.to_numpy()>1)
    assert near.shape[1]==len(SOURCES)
    expected=dict(row=row,y=y,fold=fold,ref=ref)
    families={'booking_bills':bills,'neighbors':near}
    families.update({k:cached(seed,k,v,expected) for k,v in {**PAST,**LATEST}.items()})
    return fold,base,ref,families


def replay(row,y,frame,hard):
    metrics,history={},{}
    for seed in [*map(str,SEEDS),'v131']:
        _,base,ref,families=family_scores(seed,row,y,frame)
        # Each family gets one vote. A tied internal panel cannot challenge the reference.
        agree=np.column_stack([((scores>=.5)!=ref[:,None]).mean(1)>.5 for scores in families.values()])
        count=agree.sum(1)
        baseline=ref==y
        active=ref==(base>=.5)
        metrics[seed]={}
        for name,minimum in [('majority',7),('three_quarters',10),('unanimous',13)]:
            hit=baseline^(active&(count>=minimum))
            metrics[seed][name]=dict(**paired(hit,baseline),hard_hits=int((hard&hit).sum()))
            if seed!='v131':
                history.setdefault(name,[]).append(hit)
    stable={name:int((hard&np.stack(parts).all(0)).sum()) for name,parts in history.items()}
    return dict(plans=metrics,stable_hard=stable)


def panel(row,y,frame):
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        old=z['votes'].transpose(1,0,2).reshape(len(row),-1)
    direction=old[:,0]
    all_old=np.all(old==direction[:,None],1)
    scores,refs,records=[],[],[]
    for s in map(str,SEEDS):
        fold,base,ref,families=family_scores(s,row,y,frame)
        with np.load(ART/f'nov23_joint_observed_state/{s}.npz') as z:
            for key,val in [('row',row),('y',y),('fold',fold)]:
                np.testing.assert_array_equal(z[key],val)
            bank=read_bank(row,y,fold,s).reshape(len(y),-1)
            fields=np.column_stack([z[k] for k in ('sleep_conditional','sleep_direct','basket_conditional','basket_direct')])
        all_old &= np.all((np.c_[bank,fields]>=.5)==direction[:,None],1)
        scores.append(base)
        refs.append(ref)
        records.append({k:(p>=.5)!=direction[:,None] for k,p in families.items()})
    stable={k:np.all([r[k] for r in records],axis=0).any(1) for k in records[0]}
    freq=np.stack([np.column_stack(list(r.values())).sum(1) for r in records]).mean(0)
    full_size=sum(v.shape[1] for v in records[0].values())
    return all_old,direction,np.stack(scores),np.stack(refs),stable,freq,full_size


def booking_blocks(groups,eligible,fields,risk):
    index=defaultdict(list)
    for i in np.flatnonzero(eligible):
        index[groups[i]].append(i)
    classes=defaultdict(list)
    for ids in index.values():
        ids.sort(key=lambda i:(tuple(fields[i]),risk[i],i))
        signature=tuple(tuple(v) for v in fields[ids])
        classes[signature].append(ids)
    return [np.asarray(bucket,dtype=int) for bucket in classes.values()]


def simulate(blocks,errors,matrix,reps,seed):
    # Under the conditional null, the joint outcomes of two matched bookings are exchangeable.
    # The expert identities, joint dissent and all foldwise decision counts stay intact.
    means=np.zeros(matrix.shape[1],dtype=float)
    simulated=np.zeros((reps,matrix.shape[1]),dtype=int)
    movable=np.zeros(len(errors),bool)
    rng=np.random.default_rng(seed)
    for ids in blocks:
        e=errors[ids]
        v=matrix[ids]
        means+=np.einsum('i,bic->c',e.mean(0),v)
        if len(ids)==1 or np.all(e==e[0]):
            simulated+=np.einsum('bi,bic->c',e.astype(int),v)
            if len(ids)>1:
                movable[ids]=True
            continue
        movable[ids]=True
        for start in range(0,reps,100):
            count=min(100,reps-start)
            perm=np.argsort(rng.random((count,len(ids))),axis=1)
            simulated[start:start+count]+=np.einsum('rbi,bic->rc',e[perm].astype(int),v)
    return means,simulated,movable


def main():
    row,y,frame,groups,_=load()
    _,_,hard,right=hard_cohorts(row,y)
    eligible,direction,p,refs,stable,freq,n_expert=panel(row,y,frame)
    assert len(stable)==13
    errors=(direction!=y)&eligible
    np.testing.assert_array_equal(hard,errors)
    np.testing.assert_array_equal(right,eligible&~errors)
    old=np.any([v for k,v in stable.items() if k not in LATEST],axis=0)
    count=np.sum(list(stable.values()),axis=0)
    rules={'any':count>0,'prior':old,'latest_only':(count>0)&~old,
           'one_family':count==1,'two_families':count==2,'three_or_more':count>=3,**stable}
    matrix=np.column_stack(list(rules.values())).astype(int)*eligible[:,None]
    hits=errors.astype(int)@matrix
    assert hits[:3].tolist()==[87,78,9]
    keys=list(rules)
    actual={k:dict(fixed=int(f),harmed=int(n-f)) for k,f,n in zip(keys,hits,matrix.sum(0))}
    spend=np.select([frame[BILLS].isna().any(axis=1),frame[BILLS].gt(0).sum(axis=1)==0,
                     frame[BILLS].gt(0).sum(axis=1)==1],[0,1,2],default=3)
    risk=np.where(direction,1-p.mean(0),p.mean(0))
    bounds=np.quantile(risk[eligible],np.linspace(0,1,11)[1:-1])
    bins=np.searchsorted(bounds,risk)
    # Exact history of incumbent decisions (in addition to the historical unanimous direction).
    fixed=np.c_[direction,refs.T,bins].astype(int)
    results={}
    for k,other in [('risk',[]),('risk_spending',[spend]),
                    ('risk_spending_dissent',[spend,np.digitize(freq,[.01,1.5,4.5,10.5])])]:
        features=np.column_stack([fixed,*other])
        blocks=booking_blocks(groups,eligible,features,risk)
        means,null,movable=simulate(blocks,errors,matrix,3000,661)
        assert np.allclose(means,null.mean(0),atol=.9)
        results[k]=dict(exchangeable_people=int(movable.sum()),exchangeable_errors=int((movable&hard).sum()),
                        exchangeable_rescues=int((movable&hard&rules['any']).sum()),rules={})
        for j,key in enumerate(keys):
            sample=null[:,j]
            results[k]['rules'][key]=dict(expected=float(means[j]),observed_minus_expected=float(hits[j]-means[j]),
                null95=np.quantile(sample,[.025,.975]).tolist(),upper_tail=float((1+(sample>=hits[j]).sum())/(1+len(sample))))
    exact=int(np.count_nonzero(eligible))
    write(REPORT/'placebo.json',dict(passengers=len(y),eligible=exact,hard=int(hard.sum()),right=int(right.sum()),
        experts=n_expert,families=len(stable),draws=3000,observed=actual,tests=results,
        single_replay_rules=replay(row,y,frame,hard)))
    print(actual)
    for k,m in results.items():
        print(k,m['exchangeable_people'],m['exchangeable_errors'],
              {s:m['rules'][s] for s in ('any','prior','latest_only','one_family','three_or_more')})


if __name__=='__main__':
    with threadpool_limits(1):
        main()
