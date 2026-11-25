"""Enumerate small committees on historical shared decisions and confirm their gains on held bookings."""
from itertools import combinations
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from arbitration_against_scored_nine_n26 import champions
from outside_panel_coverage_n26 import archives,projection
from target_mix_expert_routing_n26 import ROOT,batches,ticker


def packed(bits):
    return int.from_bytes(np.packbits(bits,bitorder='little').tobytes(),'little')


def search(matrix,wrong):
    error=packed(wrong)
    columns={}
    for j,col in enumerate(matrix.T):
        value=packed(col)
        if value:
            columns.setdefault(value,j)
    entries=list(columns.items())
    best={name:((0,0),(),1) for name in ('single','union','majority','agreement','any_rule')}
    def update(name,bits,indices,count):
        total=bits.bit_count()
        fixes=(bits&error).bit_count()
        quality=(2*fixes-total,-total)
        if quality>best[name][0]:
            best[name]=(quality,indices,count)
    for x,i in entries:
        update('single',x,(i,),1)
    for name in best:
        best[name]=best['single']
    for (x,i),(z,j) in combinations(entries,2):
        update('union',x|z,(i,j),1)
        update('agreement',x&z,(i,j),2)
    for (x,i),(z,j),(u,k) in combinations(entries,3):
        pair=x&z
        either=x|z
        ix=(i,j,k)
        update('union',either|u,ix,1)
        update('majority',pair|(either&u),ix,2)
        update('agreement',pair&u,ix,3)
    best['any_rule']=max(best.values(),key=lambda v:v[0])
    return best


def apply(x,recipe):
    _,cols,minimum=recipe
    if not cols:
        return np.zeros(len(x),bool)
    return x[:,cols].sum(1)>=minimum


def coverage():
    raw,y,g,fold,views,original=projection(include_full=True)
    nine,known,_,_=champions()
    from price_implied_deck_n26 import FEES
    dry=~raw[FEES].gt(0).any(axis=1).to_numpy()
    extra,_=archives(raw,y,g,fold,True)
    outside=np.column_stack([a>=.5 for a in extra.values()])
    result={}
    for tag in ('full','bookingmix','historical_to_new'):
        valid,parent=views[tag]
        assert known[valid].all()
        rows=valid&(nine==parent)
        wrong=parent!=y
        dissent=original!=parent[:,None]
        dissent[dry,-2:]=False
        main=dissent.any(1)
        ext=(outside!=parent[:,None]).any(1)
        cases={}
        for name,mask in [('both',main&ext),('only_original',main&~ext),('only_outside',~main&ext),('neither',~main&~ext)]:
            group=rows&mask
            cases[name]=dict(rows=int(group.sum()),errors=int((group&wrong).sum()))
        result[tag]=dict(joint_rows=int(rows.sum()),joint_errors=int((rows&wrong).sum()),sectors=cases)
        print(tag,result[tag],flush=True)
    return result


def run():
    cover=coverage()
    nine,known,historical,old_ids=champions()
    records={}
    for b in batches(True):
        parent=b.query_cell%2!=0
        old=b.train_cell%2!=0
        if b.tag=='full':
            assert known[b.train_ids].all()
            ref=nine
        else:
            assert np.isin(b.train_ids,old_ids).all()
            ref=historical
        t=np.flatnonzero(ref[b.train_ids]==old)
        q=nine[b.ids]==parent
        x,z=b.train_actions[t,1:100],b.query_actions[:,1:100]
        target=b.train_error[t]
        selections=search(x,target)
        actions={n:q&apply(z,v) for n,v in selections.items()}
        confirmed=[]
        for a,c in GroupKFold(2,shuffle=True,random_state=3267).split(x,groups=b.groups[t]):
            assert not np.isin(b.groups[t[a]],b.groups[t[c]]).any()
            recipe=search(x[a],target[a])['any_rule']
            verify=apply(x[c],recipe)
            good=(verify&target[c]).sum()>(verify&~target[c]).sum()
            confirmed.append(q&apply(z,recipe)&good)
        actions['one_confirmed']=confirmed[0]|confirmed[1]
        actions['two_confirmed']=confirmed[0]&confirmed[1]
        records.setdefault(b.tag,[]).append((b.ids,b.query_error,q,parent,b.strict,b.query_cell,actions))
    results={}
    for tag,items in records.items():
        ids,wrong,agree,parent,strict,cells=[np.concatenate([item[j] for item in items]) for j in range(6)]
        assert len(ids)==len(np.unique(ids)) and int((agree&wrong).sum())==cover[tag]['joint_errors']
        masks={'common_decision':agree,'both_predict_no':agree&~parent,'both_predict_yes':agree&parent,
               'common_all_applicable':agree&(cells>=4)}
        table={}
        for name in items[0][-1]:
            decision=np.concatenate([item[-1][name] for item in items])
            table[name]={k:ticker(m,wrong,decision) for k,m in masks.items()}
        results[tag]=table
        print(tag,[(n,v['common_decision']['fixed'],v['common_decision']['harmed']) for n,v in table.items()],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/common_error_committees.json').write_text(
        json.dumps(dict(coverage=cover,committees=results),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
