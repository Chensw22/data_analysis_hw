"""Measure how many shared errors can be corrected simultaneously under monotone score rules."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from sklearn.model_selection import GroupKFold

from archive_error_recovery_n23 import panel,predictions
from broad_booking_policy_n23 import EXTRA
from broad_diverse_rescue_n23 import ACTIONS,MORE
from broad_error_placement_control_n23 import SOURCES
from broad_ordered_ensemble_n23 import ART,OUT,REPORT,SEEDS,dominated,apply_rule,historical,load,write,interval


def conflicts(x,y):
    pos,neg=np.flatnonzero(y),np.flatnonzero(~y)
    a,b=np.nonzero(dominated(x[pos],x[neg]))
    return pos,neg,a,b


def solve(x,y,w,graph):
    pos,neg,a,b=graph
    n=len(y)
    edges=np.r_[np.zeros(len(pos),int),1+pos[a],1+neg]
    ends=np.r_[1+pos,1+neg[b],np.full(len(neg),n+1)]
    caps=np.r_[w[pos],np.full(len(a),1+w.sum()),w[neg]]
    capacity=coo_matrix((caps,(edges,ends)),shape=(n+2,n+2)).tocsr()
    flow=maximum_flow(capacity,0,n+1)
    residual=capacity-flow.flow
    residual.eliminate_zeros()
    assert np.all(residual.data>0)
    reached=np.zeros(n+2,bool)
    reached[breadth_first_order(residual,0,directed=True,return_predecessors=False)]=True
    good,bad=x[pos[reached[pos+1]]],x[neg[~reached[neg+1]]]
    vote=apply_rule(good,bad,x)
    assert int(w[vote!=y].sum())==flow.flow_value
    return good,bad,vote


def run():
    row,y,_,groups,_=load()
    o,t,masks=historical()
    np.testing.assert_array_equal(o,row)
    np.testing.assert_array_equal(t,y)
    prior_row,prior_y,bases,old,_,_,_=panel()
    np.testing.assert_array_equal(prior_row,row)
    np.testing.assert_array_equal(prior_y,y)
    extend={**EXTRA,**dict(list(SOURCES.items())[:4])}
    bank=np.stack([np.column_stack([old[i],predictions(row,y,s,extend)[1],predictions(row,y,s,MORE)[1]]) for i,s in enumerate(SEEDS)])
    assert bank.shape==(3,len(y),65)
    start=(bases>=.5)==y
    comparable=(((bank>=.5)==y[None,:,None]).mean(1)>=start.mean(1)[:,None]-.01).all(0)
    records,history={},{}
    for j,seed in enumerate(SEEDS):
        base,straight,fold=predictions(row,y,seed,{**ACTIONS,**MORE})
        np.testing.assert_array_equal(base,bases[j])
        x=np.column_stack([base,straight])
        small=[base]
        for folder,key in (("broad_passenger_differences/balanced","native_pair"),
                           ("broad_count_supervision","native_similar"),("broad_latent_scale","native_real")):
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                for label,expected in (("row",row),("y",y),("fold",fold)):
                    np.testing.assert_array_equal(z[label],expected)
                small.append(z[key])
        panels={"three_new_plus_base":np.column_stack(small),"direct25_plus_base":x,
                "comparable_archive_plus_base":np.column_stack([base,bank[j][:,comparable]]),
                "archive65_plus_base":np.column_stack([base,bank[j]])}
        scopes={}
        for key,fields in panels.items():
            graph=conflicts(fields,y)
            scores={}
            for cost in (1,5,len(y)+1):
                w=1+(cost-1)*masks["unresolved"].astype(int)
                _,_,vote=solve(fields,y,w,graph)
                hit=vote==y
                scores[str(cost)]=dict(accuracy=float(hit.mean()),correct=int(hit.sum()),
                    total_fixed=int((hit&~start[j]).sum()),total_harmed=int((~hit&start[j]).sum()),
                    hard_correct=int((hit&masks["unresolved"]).sum()),never_correct=int((hit&masks["never"]).sum()))
            scopes[key]=dict(columns=fields.shape[1],contradictory_pairs=len(graph[2]),optima=scores)
        # Deployment experiment restricted to direct first-stage scores. Neither
        # historical cohort nor globally selected 'comparable' status is used.
        votes={s:np.full(len(y),False) for s in ("uniform","calibration_consensus","matched_sham")}
        for f in np.unique(fold):
            ids=np.flatnonzero(fold==f)
            for a,b in GroupKFold(3,shuffle=True,random_state=823).split(ids,groups=groups[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(groups[t],groups[q]).size
                best=base[t]>=.5
                high=((straight[t]>=.5)==y[t,None]).mean(0)>=(best==y[t]).mean()-.01
                consensus=((straight[t][:,high]>=.5)==best[:,None]).all(1)
                target=consensus&(best!=y[t])
                sham=target.copy()
                rng=np.random.default_rng(694)
                for v in np.unique((10*base[t]).astype(int)):
                    c=np.flatnonzero(((10*base[t]).astype(int)==v)&(best!=y[t]))
                    sham[c]=rng.permutation(target[c])
                g=conflicts(x[t],y[t])
                for method,tag in (("uniform",np.zeros(len(t),bool)),("calibration_consensus",target),("matched_sham",sham)):
                    good,bad,_=solve(x[t],y[t],1+4*tag.astype(int),g)
                    votes[method][q]=apply_rule(good,bad,x[q])
        scored={}
        for method,v in votes.items():
            hit=v==y
            gain=hit.astype(int)-start[j]
            scored[method]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()))
            history.setdefault(method,[]).append((hit,gain))
        records[str(seed)]=dict(oracle=scopes,held_out=scored)
        np.savez_compressed(OUT/f"wide_{seed}.npz",row=row,y=y,fold=fold,**votes)
        write(REPORT/"wide_frontier.json",dict(plans=records))
        print(seed,[(k,v["optima"]["1"]["correct"],v["optima"][str(len(y)+1)]["hard_correct"]) for k,v in scopes.items()],flush=True)
    stable={}
    for k,v in history.items():
        hit=np.logical_and.reduce([h for h,_ in v])
        stable[k]={name:int((hit&m).sum()) for name,m in masks.items()}
        stable[k]["gain_ci95_pp"]=interval(np.mean([d for _,d in v],0),groups)
    write(REPORT/"wide_frontier.json",dict(plans=records,stable=stable,comparable_members=int(comparable.sum())))


if __name__=="__main__":
    run()
