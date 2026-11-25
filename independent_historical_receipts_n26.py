"""Select receipt-evidence policy on historical orders and test independent support on completely new orders."""
import json
import numpy as np
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from conditional_receipt_replacement_n26 import inputs,features,fit,swapped,compare
from full_roster_booking_validation_n23 import ROOT,data
from booking_mix_confidence_transfer_n26 import partition
from evaluate_nested_release_n26 import DECK,REVIEW,panel_predictions,stages
from held_expert_portfolios_n26 import sources,splits
from frozen_queue_shared_errors_n26 import paired
from solo_remaining_countervotes_n26 import context
from tabnet_attention_transfer_n26 import collect
from tabnet_history_test_n26 import historical_indices
from worst_booking_group_ensemble_n26 import first_stages

OUT=ROOT/'artifacts/nov26_independent_historical_receipts'
REPORT=ROOT/'reports/nov23_matched_learning_curve/independent_historical_receipts.json'


def predictions(x,group,y,t,q):
    b,tot,missing,strata,base=x
    assert not np.isin(group[t],group[q]).any()
    active=features(base,b,'active')
    full=features(base,b,'full')
    p0=fit(active[t],y[t]).predict_proba(active[q])[:,1]
    m=fit(full[t],y[t])
    original=m.predict_proba(full[q])[:,1]
    _,center,_,support,_=swapped(t,q,group,b,tot,missing,strata,base,m,True,197)
    adjustment=logit(original.clip(.001,.999))-center
    corrected=expit(logit(p0.clip(.001,.999))+adjustment)
    assert np.allclose(adjustment[support==0],0.)
    return np.column_stack([p0,corrected,support])


def policy_outputs(pred,ref):
    actions=((pred[:,:,0]>=.5)==ref[:,None]) & ((pred[:,:,1]>=.5)!=ref[:,None]) & (pred[:,:,2]>0)
    all_train,left,right=actions.T
    return dict(keep=np.zeros(len(ref),bool),pooled=all_train,first_half=left,second_half=right,
                either_half=left|right,both_halves=left&right,pooled_and_both=left&right&all_train)


def history_parent(raw,y,g,versions,v,panels):
    h=v['ids']
    votes=first_stages(raw,panels,versions)['historical']
    bank=panel_predictions(v,'historical')
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled=z['eligible']
    scope=np.zeros(len(h),bool)
    base=np.zeros(len(h),bool)
    for key,_,_,_,iq in splits(panels,g):
        if not key.startswith('historical_') or key=='historical_to_new':
            continue
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],h[iq])
            risk=np.column_stack([z['identities_error'],z['identities_outcome']])
        anchor,review,fare,now=stages(v,bank,iq,enabled[h[iq]],risk)
        np.testing.assert_array_equal(anchor,votes[iq,-1])
        # Retain only history reference decisions that had no same-order train companion.
        cold=~np.isin(g[h[iq]],g[h[v['fold']!=v['fold'][iq[0]]]])
        scope[iq]=cold&(votes[iq]==now[:,None]).all(1)&(review==now)&(fare==now)
        base[iq]=now
    return scope,base


def cache(historical,raw,y,g,h,new):
    # Exclude all later rows sharing a historical booking from every covariate computation.
    roster=np.r_[h,new]
    gi,yi=g[roster],y[roster]
    maps=np.full(len(y),-1)
    maps[roster]=np.arange(len(roster))
    x=inputs(raw.iloc[roster].reset_index(drop=True),gi)
    x_old=inputs(raw.iloc[h].reset_index(drop=True),g[h])
    for old,combined in zip(x_old,x,strict=True):
        np.testing.assert_array_equal(combined[:len(h)],old)
    half=partition(gi,'historical-receipt-independent-halves:',2)
    pred=np.full((len(roster),3,3),np.nan)
    OUT.mkdir(exist_ok=True)
    spans=[]
    for tag,tg,qg in [*historical,('new',h,new)]:
        t,q=maps[tg],maps[qg]
        assert (t>=0).all() and (q>=0).all() and not np.isin(gi[t],gi[q]).any()
        parts=[t,t[half[t]==0],t[half[t]==1]]
        assert not np.isin(gi[parts[1]],gi[parts[2]]).any()
        path=OUT/f'{tag}.npz'
        if path.exists():
            with np.load(path) as z:
                for key,val in [('t',tg),('q',qg),('train_y',y[tg]),('half',half[t])]:
                    np.testing.assert_array_equal(z[key],val)
                pred[q]=z['p']
        else:
            for i,part in enumerate(parts):
                pred[q,i]=predictions(x,gi,yi,part,q)
            np.savez_compressed(path,t=tg,q=qg,train_y=y[tg],half=half[t],p=pred[q])
        spans.append(dict(split=tag,train_rows=[len(a) for a in parts],query=len(q)))
        print(tag,spans[-1],flush=True)
    assert np.isfinite(pred).all()
    return pred[:len(h)],pred[len(h):],spans


def measure(scope,base,truth,fold,acts):
    return {k:compare(scope,np.where(a,~base,base),base,truth,fold) for k,a in acts.items()}


def run():
    raw,y,g,f,known,versions=paired()
    panels=sources()[-1]
    v=panels['historical']
    h=v['ids']
    new=np.flatnonzero(~np.isin(g,g[h]))
    assert not np.isin(g[new],g[h]).any()
    hist_parts=[]
    for k in range(3):
        t,q,_=historical_indices(raw,y,g,k)
        hist_parts.append((f'history_{k}',t,q))
    hp,npred,spans=cache(hist_parts,raw,y,g,h,new)
    valid_hist,base_hist=history_parent(raw,y,g,versions,v,panels)
    hist=measure(valid_hist,base_hist,y[h],v['fold'],policy_outputs(hp,base_hist))
    # Fix the finite choice and conservative tie break before loading later-order labels for scoring.
    choices=['keep','pooled_and_both','both_halves','pooled']
    selected=max(choices,key=lambda key:hist[key]['fixed']-hist[key]['harmed'])
    runs=collect()[0]
    items=runs['historical_to_new']
    ids,common,ref=(np.concatenate([r[j] for r in items]) for j in range(3))
    keep=common&~np.concatenate([r[3]['actual_and_scores'] for r in items])
    assert len(ids)==len(np.unique(ids)) and np.isin(ids,new).all()
    pos=np.searchsorted(new,ids)
    np.testing.assert_array_equal(new[pos],ids)
    _,_,_,_,parent,_,_,wrong,masks=context()
    hard=masks['full'][ids] & (parent[ids]==ref)
    before=measure(keep,ref,y[ids],f[ids],policy_outputs(npred[pos],ref))
    selected_cases=measure(keep&hard,ref,y[ids],f[ids],policy_outputs(npred[pos],ref))
    all_new_known=known[new]
    all_new_parent=versions[new,-1]
    any_parent=measure(all_new_known,all_new_parent,y[new],f[new],policy_outputs(npred,all_new_parent))
    report=dict(protocol='Three genuine historic validation folds, excluding every held booking from both full and half training and from donor receipts. The two hash-assigned half models share no training orders or receipt donors. The training-only pooled arm overlaps the halves and is a comparator. Both classes of comparison hold the active services, missing charges and observed total fixed. Historical rule choice maximizes net corrections among keep, both plus pooled, both, pooled; ties use the listed order. Historic scoring uses cold common parents before TabNet. Later-order evaluation uses the archived historic-to-new panel remaining after TabNet; no booking present in history can enter the final dataset.',
                samples=spans,historical=hist,selected_using_history=selected,fresh_groups=len(np.unique(g[new])),
                new_in_full_archive=any_parent,new_common_outside_tabnet=before,new_intersection_524_and_1924=selected_cases)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for n,b in [('history',hist),('fresh',before),('fresh_intersection',selected_cases),('all_new_known',any_parent)]:
        print(n,b['keep']['rows'],{k:(p['fixed'],p['harmed']) for k,p in b.items()},flush=True)
    print('selected',selected,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
