"""Bound reversals described by up to three observable passenger fields at meaningful coverage."""
import itertools
import json

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT,banks,data,latest_reference
from full_roster_six_failure_archive_n24 import prediction_sets
from full_roster_unanimous_error_profiles_n24 import groups


def build(labels,mask):
    names,indices,indptr = [],[],[0]
    family = [s.split('=')[0] for s in labels]
    for degree in (1,2,3):
        for js in itertools.combinations(range(len(labels)),degree):
            if len({family[j] for j in js})!=degree:
                continue
            present = np.flatnonzero(mask[:,js].all(1))
            if len(present)>=6:
                names.append(' & '.join(labels[j] for j in js))
                indices.extend(present)
                indptr.append(len(indices))
    return names,csr_matrix((np.ones(len(indices),np.int32),indices,indptr),shape=(len(names),len(mask)))


def study(bank,names,base,y,group,fold,eligible):
    membership = bank.T.tocsc()
    correct = base==y
    gain = np.where(correct,-1,1)
    idx = np.arange(len(y))
    def choice(rows,fraction):
        available = np.zeros(len(y),np.int32)
        available[rows]=1
        count = bank@available
        profit = bank@(available*gain)
        ok = count>=max(6,int(np.ceil(fraction*len(rows))))
        profits = np.where(ok,profit,0)
        j = int(np.argmax(profits))
        return j if profits[j]>0 else None
    result = {}
    for pct in (.02,.05):
        bound = np.zeros(len(y),bool)
        real = bound.copy()
        choices = {}
        for side in (False,True):
            use = eligible & (base==side)
            best = choice(idx[use],pct)
            if best is not None:
                ids = membership.indices[membership.indptr[best]:membership.indptr[best+1]]
                bound[ids] |= use[ids]
                choices[str(side)] = names[best]
            for level in np.unique(fold):
                ids = idx[fold==level]
                for a,b in GroupKFold(4,shuffle=True,random_state=816).split(ids,groups=group[ids]):
                    t,q = ids[a],ids[b]
                    assert not np.isin(group[t],group[q]).any()
                    rule = choice(t[use[t]],pct)
                    if rule is not None:
                        members = membership.indices[membership.indptr[rule]:membership.indptr[rule+1]]
                        real[q] |= np.isin(q,members) & use[q]
        result[str(pct)] = dict(hindsight_choices=choices)
        for name,v in [('hindsight',bound),('held_out',real)]:
            result[str(pct)][name] = dict(fixed=int((v & ~correct).sum()),harmed=int((v & correct).sum()),net=int(v@gain))
    return result


def run():
    raw,y,g,fold=data()
    _,_,_,rows,of,_,champ,_,_,_,_,_,_=aligned()
    names,h,full=banks(raw,y,fold)
    ref=latest_reference(y,g,fold,names,h,full)
    _,panels=prediction_sets(raw,y,g,fold,rows,of)
    columns,masks = groups(raw,g)
    desc,bank=build(columns,masks)
    report = {'candidate_rules':len(desc),'results':{}}
    for (tag,ix,f,base),(six,arch) in zip([('full',np.arange(len(y)),fold,ref),('historical',rows,of,champ)],panels,strict=True):
        strict = ((np.c_[six,arch]>=.5)==base[:,None]).all(1)
        result = {}
        for scope,use in [('strict',strict),('outside',~strict)]:
            result[scope]=study(bank[:,ix].tocsr(),desc,base,y[ix],g[ix],f,use)
        report['results'][tag]=result
        print(tag,result,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/descriptor_reversal.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
