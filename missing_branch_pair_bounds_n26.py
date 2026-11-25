"""Bound paired release gains over both possible upstream decisions and all eligible reviewer fits.

Run with .venv-tabpfn/bin/python. The archived panel readers also import classifiers
installed in the project's general-purpose environment.
"""
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import numpy as np
import torch
sys.path.append(str(Path(__file__).resolve().parent/'.venv/lib'/('python'+sysconfig.get_python_version())/'site-packages'))
from outside_panel_coverage_n26 import projection
from held_expert_portfolios_n26 import DECK,ROOT,REVIEW,SIX,panel_predictions,splits,sources,fare_model,six_model
from family_identity_transfer_n26 import panels_and_reference
from joint_candidate_coverage_n26 import branch_decisions
from tabicl_consensus_review_n26 import matrices
from tabicl_routed_context_n23 import DIGEST,WEIGHT
from joint_party_tabicl_n26 import model

OUT=ROOT/'artifacts/nov26_missing_branch_pairs'


def orient(v,anchor):
    # Changing the sign reverses and negates the ordered quantiles of signed log odds.
    change=v['anchor']!=anchor
    block=v['basic'].copy()
    block[change,1:4]*=-1
    block[change,4:8]=-v['basic'][change,4:8][:,::-1]
    block[:,8:]=block[:,3:8]*block[:,1:2]
    return dict(v,anchor=anchor,basic=block,relative=v['relative']*np.where(change,-1,1)[:,None],
                off=np.where(change,-v['off'],v['off']))


def run():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    raw,y,g,f,views,_=projection(include_full=True)
    compact,_,_,panels=sources()
    v=panels['full']
    p=panel_predictions(v,'full')
    _,_,_,_,names,s,_,native,_,ref=panels_and_reference()
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        prior=z['prior']
        np.testing.assert_array_equal(ref[0],z['original_prior'])
    known,current=views['full']
    np.testing.assert_array_equal(known,prior>=0)
    branches=branch_decisions(np.arange(len(y)),f,s,names,native,prior,ref[1],ref[2],
                              raw.CryoSleep.eq(False).to_numpy(),'full',891)
    nine=branches['nine']
    variants=[orient(v,a) for a in branches['all_three']]
    for b in variants:
        np.testing.assert_array_equal(b['anchor'][known],v['anchor'][known])
        np.testing.assert_array_equal(b['basic'][known],v['basic'][known])
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    deltas=[[] for _ in y]
    candidates=[[] for _ in y]
    both_wrong=[[] for _ in y]
    by_group={}
    for row in np.flatnonzero(known):
        deltas[row].append(int(current[row]==y[row])-int(nine[0,row]==y[row]))
        candidates[row].append(int(current[row]==y[row]))
        both_wrong[row].append(bool(current[row]!=y[row] and nine[0,row]!=y[row]))
    records=[]
    OUT.mkdir(exist_ok=True)
    for key,src,dst,t,q in splits(panels,g):
        if not key.startswith('full_'):continue
        k=f[q[0]]
        ask=np.flatnonzero((f==k)&~known&~np.isin(g,g[t]))
        assert not np.isin(g[ask],g[t]).any()
        if not len(ask):continue
        path=OUT/f'{key}.npz'
        if not path.exists():
            tr,held=matrices(compact,v,v,t,q)
            askx=[matrices(compact,v,b,t,ask)[1][1] for b in variants]
            risks=np.empty((2,len(ask),2))
            replay_errors=[]
            with np.load(REVIEW/f'pred_{key}.npz') as stored:
                np.testing.assert_array_equal(stored['t'],t)
                np.testing.assert_array_equal(stored['q'],q)
                for j,(name,target) in enumerate([('error',v['wrong'][t]),('outcome',y[t])]):
                    m=model(7601).fit(tr[1],target)
                    np.testing.assert_array_equal(m.classes_,[False,True])
                    est=m.predict_proba(held[1])[:,1]
                    if j:est=np.where(v['anchor'][q],1-est,est)
                    gold=stored['identities_'+name]
                    np.testing.assert_allclose(est,gold,atol=3e-5,rtol=0)
                    np.testing.assert_array_equal(est>.5,gold>.5)
                    replay_errors.append(float(abs(est-gold).max()))
                    for i,b in enumerate(variants):
                        val=m.predict_proba(askx[i])[:,1]
                        risks[i,:,j]=np.where(b['anchor'][ask],1-val,val) if j else val
            with path.with_suffix('.tmp').open('wb') as fp:
                np.savez_compressed(fp,t=t,q=q,ask=ask,labels=y[t],risks=risks,replay=replay_errors,checkpoint=DIGEST)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for n,val in [('t',t),('q',q),('ask',ask),('labels',y[t]),('checkpoint',DIGEST)]:
                np.testing.assert_array_equal(z[n],val)
            risks=z['risks']
            records.append(dict(key=key,missing=int(len(ask)),replay=float(z['replay'].max())))
        paired=[]
        for i,b in enumerate(variants):
            review=b['anchor'][ask]^(risks[i]>.5).all(1)
            fare=fare_model(review,p['neural_deck_raw'][ask],p['neural_deck_fine'][ask],enabled[ask])
            local=six_model(fare,np.column_stack([p[n][ask] for n in SIX]),v['votes'][ask,:18],enabled[ask])>=.5
            gain=(local==y[ask]).astype(int)-(nine[i,ask]==y[ask])
            paired.append(gain)
            for row,e,c in zip(ask,gain,local==y[ask],strict=True):
                deltas[row].append(e)
                candidates[row].append(c)
                both_wrong[row].append(bool(not c and nine[i,row]!=y[row]))
        for group in np.unique(g[ask]):
            pos=g[ask]==group
            assert np.array_equal(np.sort(ask[pos]),np.flatnonzero((g==group)&~known))
            values=np.asarray(paired)[:,pos]
            by_group.setdefault(group,[]).append([int(values.min(0).sum()),int(values.max(0).sum())])
        print(key,len(ask),flush=True)
    lo,hi=np.array([(min(d),max(d)) for d in deltas]).T
    clo,chi=np.array([(min(d),max(d)) for d in candidates]).T
    assert lo[known].sum()==hi[known].sum()==81 and (~known).sum()==161
    report=dict(known_net=81,missing=int((~known).sum()),missing_net_bounds=[int(z[~known].sum()) for z in [lo,hi]],
                whole_net_bounds=[int(z.sum()) for z in [lo,hi]],candidate_correct_bounds=[int(z.sum()) for z in [clo,chi]],
                additional_exact_candidate_predictions=int(((clo==chi)&~known).sum()),
                additional_exact_paired_effects=int(((lo==hi)&~known).sum()),fits=records,
                semantics='Fixed original validation training sets. On excluded queries enumerate both upstream branches and every existing same-fold reviewer whose support excludes the entire order. Per-row extrema give conservative full-roster bounds, not a realized accuracy.')
    grouped=[sum(fun(np.array(choices)[:,j]) for choices in by_group.values()) for j,fun in enumerate([min,max])]
    report['whole_net_bounds_same_order_same_fit']=[int(81+v) for v in grouped]
    report['per_fold_net_bounds'] = [
        [int(lo[f == k].sum()), int(hi[f == k].sum())] for k in np.unique(f)
    ]
    report['leave_one_fold_out_net_bounds'] = [
        [int(lo[f != k].sum()), int(hi[f != k].sum())] for k in np.unique(f)
    ]
    report['both_wrong_8532']=int(sum(both_wrong[i][0] for i in np.flatnonzero(known)))
    report['both_wrong_whole_bounds']=[int(sum(op(values) for values in both_wrong)) for op in (min,max)]
    print(report,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/missing_branch_pair_bounds.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
