"""Audit unused November 23 submission files on matched historical and whole-booking folds."""
import json
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_cross_family_submission_choices_n26 import inventory
from audit_export_score_limits_n23 import all_candidate_votes
from audit_legacy_queue_reach_n26 import NAMES as OLD, contemporary, stats
from broad_booking_canonical_n23 import load_roster
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT
from broad_service_bag_trees_n23 import expand, fit as service_tree
from broad_service_pairs_n23 import fit_pair, pair_table
from certified_tree_release_n25 import structural_inputs
from forecast_frozen_queue_n26 import NAMES as RELEASES
from full_roster_booking_validation_n23 import OUT, data
from grouped_peer_release_replay_n25 import limits
from grouped_peer_teacher_n25 import OUT as TEACHER
from pending_submission_diversity_n23 import PAIRS
from reserve_composite_replay_n26 import compose, composite_ranges
from service_pair_change_budget_n23 import changed
from six_way_against_scored_nine_n26 import test_decisions
from tabnet_attention_transfer_n26 import collect

REPORT=ROOT/'reports/nov23_matched_learning_curve/seven_reserve_releases.json'
CACHE=ART/'nov26_seven_reserve_releases'
BUDGET,TREE,PAIR=OLD[1],OLD[4],OLD[5]


def read(path):
    return json.loads(path.read_text())


def signed(pred,baseline,y,groups,fold,scope):
    report=stats(pred,baseline,y,scope,fold,groups)
    report.update(rows=int(scope.sum()),bookings=int(np.unique(groups[scope]).size),
                  gain_pp=float(report['net']*100/scope.sum()))
    return report


def files(test):
    scored=inventory(test)
    capacity=read(REPORT.with_name('queued_target_capacity.json'))
    denominator=capacity['compatible_scoring_denominator']
    assert denominator==len(test) and capacity['reference_correct']==3496 and capacity['target_correct']==3636
    queue={}
    for name in list(RELEASES.values())[1:]:
        v,sha=test_decisions(name,test)
        state=read(ART/name/'queue_status.json')
        assert state['sha256']==sha
        queue[name]=v
    exports={}
    for path in sorted(ART.glob('*/submission.csv')):
        name=path.parent.name
        manifest=ROOT/'reports'/name/'manifest.json'
        if not name.startswith('proposal_n23_') or not manifest.exists() or name in scored or name in queue:
            continue
        committed=subprocess.run(['git','show','HEAD:'+str(manifest.relative_to(ROOT))],
                                 cwd=ROOT,capture_output=True)
        if committed.returncode:
            continue
        assert committed.stdout==manifest.read_bytes()
        vote,sha=test_decisions(name,test)
        assert not list((manifest.parent/'online_attempts').glob('*.json'))
        assert not (path.parent/'queue_status.json').exists()
        assert read(path.parent/'manifest.json')['sha256']==sha
        d={r:int((vote!=record['vote']).sum()) for r,record in scored.items()}
        near=min(d.values())
        nearest=[k for k in d if d[k]==near]
        assert len(nearest)==1
        observed=scored[nearest[0]]['score']
        correct=round(observed*denominator)
        assert abs(correct/denominator-observed)<.00001
        upper=correct+near
        exports[name]=dict(sha256=sha,nearest_scored_test_distance=near,
            nearest_file_aggregate_limit=dict(denominator=denominator,correct=correct,maximum_correct=upper,
                maximum_fraction=upper/denominator,can_beat_nine=upper>capacity['reference_correct'],
                can_reach_85=upper>=capacity['target_correct']),
            nearest_scored={r:{k:scored[r][k] for k in ('score','ref','sha256','receipt')} for r in d if d[r]==near},
            test_distance_to_queue={r:int((vote!=v).sum()) for r,v in queue.items()},
            test_distance_to_nine=d[RELEASES['nine']])
    ref=scored['proposal_v131_party_consensus']['vote']
    def shifted(name):
        return test_decisions('proposal_n23_'+name,test)[0]!=ref
    order_ids=test.str.split('_').str[0]
    size=order_ids.map(order_ids.value_counts()).to_numpy()
    composed=compose(OLD,shifted('personal_bill_quarter')|shifted('three_corrections_mean'),
        shifted('service_bag_tree_quarter'),shifted('service_bag_quarter'),shifted('service_pair_zero_donor'),
        shifted('service_pair_quarter'),shifted('service_pair_change_150'),shifted('registration_tree_quarter'),size)
    for n,v in composed.items():
        np.testing.assert_array_equal(test_decisions(n,test)[0],ref^v)
    assert set(exports)==set(OLD)
    return exports,len(scored)


def grouped_base(raw,test,y,fold):
    patch,peer,sleep,party=structural_inputs(pd.concat([raw,test],ignore_index=True))
    p108=np.full(len(y),np.nan)
    location=ART/'nov26_patched_cabins'
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId)
        witness,family=z['witness'],z['family']
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(location/f'fold{k}.npz') as z,np.load(TEACHER/f'release{k}.npz') as old:
            for key,v in [('q',q),('t',t),('train_y',y[t]),('patch_ids',q[patch[q]])]:
                np.testing.assert_array_equal(z[key],v)
            repaired=z['native'].copy()
            repaired[patch[q]]=z['patch_p']
            p108[q],_,_=limits(repaired,np.zeros(len(q),bool),peer[q],old['shift'],sleep[q],witness[q],party[q],family[q])
            np.testing.assert_array_equal(p108[q][~patch[q]],old['p108'][~patch[q]])
    with np.load(location/'known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['row'],np.arange(len(y)))
        np.testing.assert_array_equal(z['fold'],fold)
        prior=z['prior']
    return p108,prior


def tree_probabilities(frame,y,g,fold):
    owner,records,prior=expand(frame)
    ids=np.arange(len(y))
    buyer,pair_records,weight=pair_table(owner,records,prior,ids,ids)
    votes=np.empty((len(y),2))
    CACHE.mkdir(exist_ok=True)
    for k in np.unique(fold):
        q,t=ids[fold==k],ids[fold!=k]
        assert not np.isin(g[q],g[t]).any()
        path=CACHE/f'fold{k}.npz'
        if not path.exists():
            held=np.isin(owner,q)
            tree=service_tree(records[~held],y[owner[~held]],prior[~held])
            p=tree.predict_proba(records[held],thread_count=2)[:,1]
            bag=np.bincount(owner[held],weights=p*prior[held],minlength=len(y))[q]
            held=np.isin(buyer,q)
            pair=fit_pair(pair_records[~held],y[buyer[~held]],weight[~held])
            p=pair.predict_proba(pair_records[held],thread_count=2)[:,1]
            both=np.bincount(buyer[held],weights=p*weight[held],minlength=len(y))[q]
            np.savez_compressed(path,q=q,t=t,train_y=y[t],bag=bag,pair=both)
        with np.load(path) as z:
            for key,value in [('q',q),('t',t),('train_y',y[t])]:
                np.testing.assert_array_equal(z[key],value)
            votes[q]=np.column_stack([z['bag'],z['pair']])
        print('verified grouped service fits',k,flush=True)
    return votes


def main():
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda k:k!='Name')
    records,total=files(test.PassengerId)
    row,y,frame,groups,_=load()
    w=read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    ref,old=all_candidate_votes(frame,groups,row,y,'v131',w)
    comparisons,fold,cold=contemporary(row,ref,groups)
    legacy=read(REPORT.with_name('legacy_queue_reach.json'))
    for n in OLD:
        for scope,m in [('historical',np.ones(len(y),bool)),('historical_held_orders',cold)]:
            records[n][scope]={s:signed(old[n],p,y,groups,fold,m) for s,p in comparisons.items()}
            source=legacy['candidates'][n.removeprefix('proposal_n23_')]['all_historical' if scope=='historical' else 'whole_order_cold']['versus_nine']
            for k,v in source.items():assert records[n][scope]['nine'][k]==v
    raw,y,g,f=data()
    results,truth,gr,ff,current=collect()
    for a,b in [(truth,y),(g,gr),(f,ff)]:np.testing.assert_array_equal(a,b)
    compared=dict(zip(list(RELEASES)[:-1],current.T,strict=True))
    known=np.zeros(len(y),bool)
    tabnet=known.copy()
    for q,common,parent,act in results['full']:
        assert not known[q].any()
        known[q]=True
        tabnet[q]=parent^(common&act['actual_and_scores'])
    compared['tabnet']=tabnet
    p108,prior=grouped_base(raw,test,y,f)
    np.testing.assert_array_equal(known,prior>=0)
    with np.load(OUT/'budget.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('fold',f),('y',y)]:np.testing.assert_array_equal(z[k],v)
        budget=z['p']
    frame,rg,size,_=load_roster()
    np.testing.assert_array_equal(rg[:len(y)],g.astype(int))
    frame['BookingSize']=size
    tree,pair=tree_probabilities(frame.iloc[:len(y)].copy(),y,g,f).T
    grouped={n:changed(p,p108,prior==1,weight) for n,p,weight in [(BUDGET,budget,.25),(TREE,tree,.25),(PAIR,pair,w)]}
    fresh=~np.isin(g,g[row])
    for n,v in grouped.items():
        for s,m in [('grouped_eight',known),('grouped_eight_new_order',known&fresh)]:
            records[n][s]={key:signed(v,baseline,y,g,f,m) for key,baseline in compared.items()}
    composite=composite_ranges(frame,y,g,f,p108,prior,{**grouped,'pair_probability':pair},OLD,CACHE)
    for n,endpoints in composite.items():
        left=(prior==1)^endpoints['low']
        right=(prior==1)^endpoints['high']
        for s,scope in [('grouped_eight',known),('grouped_eight_new_order',known&fresh)]:
            resolved=scope&(left==right)
            missing=scope&~resolved
            by_model={}
            for key,baseline in compared.items():
                item=signed(left,baseline,y,g,f,resolved)
                # Any unresolved individual's prediction may take either side, yielding
                # a conservative interval for total net correct on the full named scope.
                item['unresolved_rows']=int(missing.sum())
                item['all_scope_net_bounds']=[item['net']-int((missing&(baseline==y)).sum()),
                                               item['net']+int((missing&(baseline!=y)).sum())]
                by_model[key]=item
            records[n][s+'_determined']=by_model
    report=dict(scored_files=total,queued_files=len(RELEASES)-1,separate_folds='Historical uses 6,954 earlier passengers including a strict same-order holdout subset. Grouped eight refits three deployable contenders using the exact repaired v108 probabilities, the known v131 sign, and the original blend weights. Both eightfold scopes exclude 161 unresolved parent cases. New orders excludes any order in the historical sample. For the four composite files replay the service, individual and registration models, bound the zero-donor ranking with unresolved parent decisions, and permit both values for the undocumented nested three-way vote. Only rows with a unique resulting decision enter repair/harm counts; all-scope net bounds also include the indeterminate portion. The test distances and aggregate score limits use no passenger labels.',
                comparisons='Historical scores compared only with nine and eight individually replayed queue branches; filtered and TabNet have no verified same historical query decisions. All ten queue entries are directly compared on exact eightfold evidence for three releases. Distance to a scored CSV is unlabeled and does not estimate the reserve score.',
                records=records)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for n,r in records.items():
        print(n,[(s,r[s]['nine']['net'],r[s]['nine']['net_by_fold']) for s in ('historical_held_orders','grouped_eight','grouped_eight_new_order') if s in r],r['nearest_scored_test_distance'],list(r['nearest_scored']),flush=True)


if __name__=='__main__':
    with threadpool_limits(2):main()
