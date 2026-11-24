"""Audit archived models on the remaining common mistakes under both validation designs."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from full_roster_champion_overlap_n24 import aligned
from full_roster_conditional_pool_n23 import ROOT, data, banks, latest_reference
from full_roster_gate_probability_n23 import gate_probabilities
from full_roster_adaptive_bill_prior_n23 import EVIDENCE
from full_roster_booking_validation_n23 import FAMILIES
from full_roster_kernel_repair_overlap_n24 import from_cache
from full_roster_reference_innovation_n24 import bank

def fixes(a, b, y):
    return [int(np.sum((a == y) & (b != y))), int(np.sum((a != y) & (b == y)))]
def audit():
    raw,y,g,fold = data()
    names,half,full = banks(raw,y,fold)
    ref=latest_reference(y,g,fold,names,half,full)
    gate,_=gate_probabilities(names,half,full,y,g,fold)
    np.testing.assert_array_equal(ref,gate>=.5)
    _,oy,og,rows,of,_,champ,_,_,_,_,old,_=aligned()
    with np.load(ROOT/'artifacts/nov24_remaining_cases/audit.npz') as z:
        for k,v in [('passenger',raw.PassengerId),('y',y),('fold',fold)]:
            np.testing.assert_array_equal(z[k],v)
        scores=z['scores']
        np.testing.assert_array_equal(scores[:,:19],full)
        _,prev=from_cache('full',np.arange(len(y)),fold,y,g)
        np.testing.assert_array_equal(scores[:,19:23],prev)
        votes=scores>=.5
        uni=(votes==votes[:,[0]]).all(1)
        bad=uni & (votes[:,0]!=y)
        np.testing.assert_array_equal(bad,z['missed'])
        assert votes.shape==(8693,26) and bad.sum()==223
    specs={
    'bill_regime_tree':('p','pooled,constant,registration','t','q','truth'),
    'class_neighborhood_mass':('p','personal,density,mixed,shuffled','t','q','truth'),
    'companion_transition':('scores','personal,real,removed,swapped','train','query','train_labels'),
    'error_partition_kernel':('prob','rand,all_grad,mistake,matched_random','train','query','truth'),
    'labelled_reference_pairs':('p','point,joint,near_sep,near_rel,global_rel','t','q','truth'),
    'permutation_selection':('prob','all,selected,random,protected_select,protected_random','train','query','labels'),
    'recursive_metric':('prob','isotropic,diag1,diag2,full1,full2','train','query','truth'),
    'service_order_mixture':('p','independent,empty,joint','train','query','truth')}
    skip={'class_neighborhood_mass/mixed','error_partition_kernel/mistake','labelled_reference_pairs/near_rel','recursive_metric/full1','service_order_mixture/independent','service_order_mixture/joint'}
    splits=[('historical',rows,of),('full',np.arange(len(y)),fold)]
    out={}
    def read(folder,key,tags,tk,qk,yk,mode=None):
        fields=[]
        for tag,idx,s in splits:
            p=np.full((len(idx),len(tags)),np.nan)
            for f in np.unique(s):
                q=idx[s==f];t=idx[(s!=f)&~np.isin(g[idx],g[q])]
                assert not np.isin(g[t],g[q]).any()
                fname=f'{tag}_{f}.npz' if mode is None else f'{mode}_{"eightfold" if tag=="full" else tag}_{f}.npz'
                with np.load(ROOT/f'artifacts/nov24_{folder}'/fname) as z:
                    for zk,v in [(qk,q),(tk,t),(yk,y[t])]:
                        np.testing.assert_array_equal(z[zk],v)
                    p[s==f]=z[key] if mode is None else z[key].mean(1)[:,None]
            fields.append(p)
        for j,name in enumerate(tags):
            k=f'{folder}/{name}'
            if k not in skip:
                out[k]=(fields[0][:,j],fields[1][:,j])
    for folder,(key,mods,tk,qk,yk) in specs.items():
        read(folder,key,mods.split(','),tk,qk,yk)
    for mode in ['linear','relu','maxout']:
        read('maxout','pair',[mode],'t','q','y',mode)
    for mode in ['independent','truncated','reported_sleep']:
        with np.load(ROOT/f'artifacts/nov24_zero_truncated/{mode}.npz') as z:
            for k,v in [('rows',rows),('oldfold',of),('fold',fold),('passenger',raw.PassengerId)]:
                np.testing.assert_array_equal(z[k],v)
            out['zero_truncated/'+mode]=(z['historical'],z['full'])
    for a,b in out.values():
        for p in [a,b]:
            assert np.isfinite(p).all() and (0<=p).all() and (p<=1).all()
    hits=np.stack([b>=.5 for a,b in out.values()],axis=-1)==y[:,None]
    remaining = bad & ~hits.any(axis=1)
    folder = ROOT/'artifacts/nov24_remaining_cases'
    np.savez_compressed(folder/'archive_scores.npz', row=rows, old_fold=of, fold=fold,
                        passenger=raw.PassengerId.to_numpy(str), y=y, names=list(out),
                        historical=np.column_stack([a for a,b in out.values()]),
                        full=np.column_stack([b for a,b in out.values()]))
    cache = folder/'after_archive.npz'
    np.savez_compressed(cache, passenger=raw.PassengerId.to_numpy(str), y=y, fold=fold,
                        missed=remaining, baseline_misses=bad)
    rh=bad[rows] & (champ!=oy)
    seen=~np.isin(g,og)
    result={'model_count':len(out),'source_directories':sorted({k.split('/')[0] for k in out}),
            'hard_union':int((bad & hits.any(1)).sum()), 'historical_v131_also_wrong':int(rh.sum()),
            'distinct_prediction_vectors':int(len({b.tobytes() for _,b in out.values()})),'models':{}}
    for j,(k,(a,b)) in enumerate(out.items()):
        ob,fb=a>=.5,b>=.5
        active=uni & (ref==votes[:,0])
        bo=np.where(champ==(old[:,0]>=.5),(.75*old[:,0]+.25*a)>=.5,champ)
        bn=(.75*gate+.25*b)>=.5
        stable=rh & (ob==oy) & (fb[rows]==oy)
        result['models'][k]={'on223':int(np.sum(bad&hits[:,j])), 'only_this_variant':int(np.sum(bad&hits[:,j]&(hits.sum(1)==1))),
            'repeated_on177':int(stable.sum()), 'unanimous_fixes_harms':fixes(np.where(active,fb,ref),ref,y),
            'hist_direct':fixes(ob,champ,oy),'full_direct':fixes(fb,ref,y),
            'hist_quarter':fixes(bo,champ,oy),'full_quarter':fixes(bn,ref,y),'quarter_new_orders':fixes(bn[seen],ref[seen],y[seen]),
            'quarter_on223':int((bad&(bn==y)).sum()),'on223_by_target':np.bincount(y[bad&hits[:,j]],minlength=2).tolist()}
    a,b=out['permutation_selection/selected']
    flip=uni & ((b>=.5)!=ref) & (ref==votes[:,0])
    result['five_dissent']=[]
    for i in np.flatnonzero(flip):
        p=np.flatnonzero(rows==i)
        result['five_dissent'].append(dict(row=int(i),passenger=raw.PassengerId.iat[i],full_fold=int(fold[i]),
            full_prob=float(b[i]),historical_prob=a[p].tolist(),historical_incumbent=champ[p].tolist(),
            transported=bool(y[i]),belongs_to_1044=bool(seen[i])))
    # Exact matched historical panel available from 9 native and 7 new scores.
    original=np.full((len(rows),9),np.nan)
    for f in np.unique(of):
        q=rows[of==f]; t=rows[(of!=f)&~np.isin(g[rows],g[q])]
        with np.load(ROOT/f'artifacts/nov24_native_prior_gate/fold_{f}_with_provenance.npz') as z:
            for k,v in [('query',q),('new_train',t),('historical_train',rows[of!=f]),('y',y[t])]:
                np.testing.assert_array_equal(z[k],v)
            original[of==f]=z['scores']
    short=[names.index(k) for k in [*FAMILIES,'mitra96',*EVIDENCE]]
    _,pr=from_cache('historical',rows,of,y,g)
    ext=bank(raw,y,g,rows,of,'historical')
    h16=np.column_stack([original,pr,ext])>=.5
    f16=np.column_stack([full[:,short],scores[:,19:]])>=.5
    compare={}
    for tag,ball,candidate,base,target in [('historical',h16,a,champ,oy),('full',f16,b,ref,y)]:
        exposed=(ball==base[:,None]).all(1)
        rule=np.where(exposed,candidate>=.5,base)
        compare[tag]={'available':int(exposed.sum()),'fixed_harmed':fixes(rule,base,target)}
    old_flip=(h16==champ[:,None]).all(1) & ((a>=.5)!=champ)
    new_flip=(f16==ref[:,None]).all(1) & ((b>=.5)!=ref)
    compare['same_row_flipped_in_both']=int(np.sum(old_flip & new_flip[rows]))
    compare['same_row_fixed_in_both']=int(np.sum(old_flip & (champ!=oy) & new_flip[rows] & (ref[rows]!=oy)))
    compare['full_1044']=fixes(np.where(new_flip,b>=.5,ref)[seen],ref[seen],y[seen])
    result['matched_16_selected']=compare
    result['practical_panels'] = {}
    for size in (3,5,9):
        if size == 3:
            positions = [short.index(names.index(k)) for k in ('mitra96','v44_without_name','tabicl')]
        else:
            positions = list(range(size))
        report = {}
        for tag, ballot, baseline, pred, target in [('historical',h16,champ,a,oy),('full',f16,ref,b,y)]:
            enabled = (ballot[:,positions] == baseline[:,None]).all(1)
            decision = np.where(enabled,pred>=.5,baseline)
            report[tag] = dict(available=int(enabled.sum()),fixed_harmed=fixes(decision,baseline,target))
            if tag == 'full':
                report['unseen'] = fixes(decision[seen],baseline[seen],target[seen])
        result['practical_panels'][str(size)] = report
    path = ROOT/'reports/nov23_matched_learning_curve/archive_remaining_cases.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(result['model_count'],result['hard_union'],result['matched_16_selected'],result['practical_panels'])
if __name__=='__main__':
    with threadpool_limits(2):
        audit()
