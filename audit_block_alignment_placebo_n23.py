"""Check whether the class-alignment transfer selects unseen recurring errors beyond its scope."""
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from audit_stable_rescue_placebo_n23 import booking_blocks,panel,simulate
from broad_geographic_transfer_n23 import BILLS,load
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write
from broad_tree_field_dividends_n23 import hard_cohorts

REPORT=ROOT/'reports/nov23_block_alignment_placebo'


def alignment(row,y,scores,refs,direction):
    margin={'authentic':[],'sham':[]}
    for s,base,ref in zip(map(str,SEEDS),scores,refs):
        with np.load(ART/f'nov23_manifest_block_dependence/{s}.npz') as z:
            for k,val in [('row',row),('y',y),('ref',ref),('base',base)]:
                np.testing.assert_array_equal(z[k],val)
            odds=lambda p:logit(p.clip(.002,.998))
            for k,h in margin.items():
                d=odds(z[k+'_real'])-odds(z[k+'_synthetic'])
                v=np.where(ref==(base>=.5),odds(base)+d,np.where(ref,30.,-30.))
                h.append(np.where(direction,-v,v))
    return {k:np.stack(h).min(0) for k,h in margin.items()}


def main():
    row,y,frame,groups,_=load()
    eligible,direction,p,refs,stable,_,_=panel(row,y,frame)
    old=np.any(list(stable.values()),axis=0)
    hard=(direction!=y)&eligible
    correct=(direction==y)&eligible
    _,_,known_hard,known_right=hard_cohorts(row,y)
    np.testing.assert_array_equal(hard,known_hard)
    np.testing.assert_array_equal(correct,known_right)
    margins=alignment(row,y,p,refs,direction)
    real=margins['authentic']>0
    sham=margins['sham']>0
    assert (real&hard).sum()==97 and (old&hard).sum()==87
    risk=np.where(direction,1-p.mean(0),p.mean(0))
    bins=np.searchsorted(np.quantile(risk[eligible],np.arange(1,10)/10),risk)
    fixed=np.c_[direction,refs.T,bins].astype(int)
    bills=frame[BILLS]
    categories=['missing','zero','one','multiple']
    spend=np.select([bills.isna().any(axis=1),bills.gt(0).sum(axis=1)==0,
                     bills.gt(0).sum(axis=1)==1],[0,1,2],default=3)
    primary=np.column_stack([fixed,spend])
    _,codes=np.unique(primary,axis=0,return_inverse=True)
    equal_sham=np.zeros(len(y),bool)
    for b in np.unique(codes[eligible]):
        ix=np.flatnonzero(eligible&(codes==b))
        n=real[ix].sum()
        order=np.lexsort((row[ix],-margins['sham'][ix]))
        equal_sham[ix[order[:n]]]=True
    base_rules={'old':old,'real':real,'union':old|real,'overlap':old&real,'novel':~old&real,
                'sham':sham,'equal_budget_sham':equal_sham,'sham_novel':equal_sham&~old}
    scopes={}
    for j,name in enumerate(categories):
        for value in [False,True]:
            scopes[name+('_to_negative' if value else '_to_positive')]=(spend==j)&(direction==value)
    selected={**base_rules,**{k:s&real for k,s in scopes.items()}}
    keys=list(selected)
    mtx=eligible[:,None]*np.column_stack(list(selected.values())).astype(int)
    hits=hard.astype(int)@mtx
    counts={k:dict(fixed=int(v),harmed=int(q-v)) for k,v,q in zip(keys,hits,mtx.sum(0))}
    people={}
    for key,mask in scopes.items():
        people[key]=dict(old_errors=int((hard&mask).sum()),old_correct=int((correct&mask).sum()),
                         novel=int((hard&mask&~old&real).sum()),novel_harms=int((correct&mask&~old&real).sum()))
    planet=np.unique(frame.HomePlanet,return_inverse=True)[1]
    sleep=np.unique(frame.CryoSleep,return_inverse=True)[1]
    zero_breakdown=[]
    for origin in np.unique(frame.HomePlanet):
        for sleeping in np.unique(frame.CryoSleep):
            scope=scopes['zero_to_negative'] & frame.HomePlanet.eq(origin).to_numpy() & frame.CryoSleep.eq(sleeping).to_numpy()
            selected=scope&real
            if np.any(eligible&selected):
                zero_breakdown.append(dict(origin=str(origin),sleep=str(sleeping),fixed=int((hard&selected).sum()),
                                           harmed=int((correct&selected).sum()),novel_fixed=int((hard&selected&~old).sum()),
                                           equal_sham_fixed=int((scope&hard&equal_sham).sum()),
                                           equal_sham_harmed=int((scope&correct&equal_sham).sum())))
    tests={}
    for name,fields in [('risk_spending',primary),('prior_dissent',np.c_[primary,old]),
                        ('sleep_origin',np.c_[primary,old,sleep,planet])]:
        blocks=booking_blocks(groups,eligible,fields,risk)
        mean,draws,movable=simulate(blocks,hard,mtx,3000,661)
        info={}
        for j,k in enumerate(keys):
            info[k]=dict(expected=float(mean[j]),excess=float(hits[j]-mean[j]),
                         null95=np.quantile(draws[:,j],[.025,.975]).tolist(),
                         upper_tail=float((1+(draws[:,j]>=hits[j]).sum())/(1+len(draws))))
        r,d=keys.index('real'),keys.index('equal_budget_sham')
        dif=draws[:,r]-draws[:,d]
        true=int(hits[r]-hits[d])
        tests[name]=dict(movable_people=int(movable.sum()),movable_hard=int((movable&hard).sum()),
                         movable_real_rescues=int((movable&hard&real).sum()),rules=info,
                         real_minus_equal_sham=dict(observed=true,expected=float(mean[r]-mean[d]),
                                                    null95=np.quantile(dif,[.025,.975]).tolist(),
                                                    upper_tail=float((1+(dif>=true).sum())/(1+len(dif)))))
    assert counts['real']['fixed']+counts['real']['harmed']==counts['equal_budget_sham']['fixed']+counts['equal_budget_sham']['harmed']
    assert sum(v['novel'] for v in people.values())==counts['novel']['fixed']
    write(REPORT/'alignment_placebo.json',dict(observed=counts,scopes=people,zero_details=zero_breakdown,tests=tests))
    print(counts,flush=True)
    for k,v in tests.items():
        print(k,v['movable_people'],v['movable_hard'], {q:v['rules'][q] for q in ['real','novel']},v['real_minus_equal_sham'])


if __name__=='__main__':
    with threadpool_limits(1):
        main()
