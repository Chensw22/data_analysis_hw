"""Average stop/split decisions over all orders of observed booking and spending contexts."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from archive_error_recovery_n23 import ART, ROOT, SEEDS, panel, predictions
from broad_booking_policy_n23 import EXTRA
from broad_booking_policy_replay_n23 import interval
from broad_error_placement_control_n23 import SOURCES
from broad_geographic_transfer_n23 import BILLS, load
from group_crossfit_residual_discovery_n23 import teachers
from relational_error_corrector import write

OUT = ART/"nov23_broad_partition_evidence"
REPORT = ROOT/"reports/nov23_broad_partition_evidence"


def historical():
    row,y,base,old,*_ = panel()
    members = {**EXTRA,**dict(list(SOURCES.items())[:4])}
    recent = np.stack([predictions(row,y,s,members)[1] for s in SEEDS])
    hit = (np.concatenate([old,recent],axis=2)>=.5)==y[None,:,None]
    ref = (base>=.5)==y
    repeat = ~ref.any(0)
    masks = dict(ever=~ref.all(0), repeat=repeat, unresolved=repeat&~hit.all(0).any(1),
                 never=repeat&~hit.any((0,2)))
    assert [m.sum() for m in masks.values()]==[1518,1070,509,172]
    return row,y,masks


def partition_axes(frame):
    def band(values,cuts):
        return np.where(np.isfinite(values),np.digitize(values,cuts),-1)
    def joint(*values):
        return pd.factorize(pd.MultiIndex.from_arrays(values))[0]
    bills=frame[BILLS]
    risk=bills[["Spa","VRDeck","RoomService"]].sum(axis=1,min_count=1)
    other=bills[["FoodCourt","ShoppingMall"]].sum(axis=1,min_count=1)
    total=risk.fillna(0)+other.fillna(0)
    dominance=np.where(total>0,bills.fillna(-1).to_numpy().argmax(1)+1,0)
    context=[frame.CryoSleep,frame.HomePlanet,joint(frame.Deck,frame.Side),joint(frame.Destination,frame.VIP),
             joint(band(frame.Age,[13,22,40]),band(frame.BookingSize,[2,4])),
             joint(band(risk,[1,500,1500]),band(other,[1,500,1500])),
             joint(dominance,band(frame.CabinNumber,[300,800,1300]))]
    amounts=[band(bills[n],[1,400,1100]) for n in BILLS]
    return dict(context=context,services=[joint(frame.CryoSleep,frame.HomePlanet),joint(frame.Side,frame.Deck),*amounts])


class Partitions:
    def __init__(self,axes):
        self.n=len(axes)
        size=1<<self.n
        self.code=[np.zeros(len(axes[0]),int)]
        self.first=[np.array([0])]
        self.child={}
        for mask in range(1,size):
            bit=mask&-mask
            col=pd.factorize(axes[bit.bit_length()-1])[0]
            raw=self.code[mask^bit]*(col.max()+2)+col+1
            _,index,inv=np.unique(raw,return_inverse=True,return_index=True)
            self.code.append(inv)
            self.first.append(index)
        for mask in range(size):
            self.child[mask]=[mask|1<<j for j in range(self.n) if not mask&(1<<j)]

    def infer(self,t,q,y,train_offset,query_offset,scale):
        # Discrete prior over leaf-specific log-odds shifts, integrated exactly.
        shifts=np.arange(-8,9)*(.375*scale)
        logprior=-(shifts/scale)**2/2
        logprior-=logsumexp(logprior)
        z=train_offset[:,None]+shifts
        ll=y[t,None]*z-np.logaddexp(0,z)
        query_prob=expit(query_offset[:,None]+shifts)
        mass,pred={},{}
        root_shallow=None
        for mask in reversed(range(len(self.code))):
            code=self.code[mask]
            n=len(self.first[mask])
            cell=code[t]
            scores=np.stack([np.bincount(cell,weights=l,minlength=n) for l in ll.T],axis=1)+logprior
            leaf=logsumexp(scores,axis=1)
            leaf_prob=(query_prob*np.exp(scores-leaf[:,None])[code[q]]).sum(1)
            if not self.child[mask]:
                mass[mask],pred[mask]=leaf,leaf_prob
                continue
            branch=[np.bincount(code[self.first[c]],weights=mass[c],minlength=n) for c in self.child[mask]]
            odds=np.stack([leaf,*[v-np.log(len(branch)) for v in branch]])
            if mask==0:
                # Every one-field model has equal prior probability. This
                # control forbids the recursive interactions of the full model.
                weights=[]
                estimates=[]
                for c in self.child[0]:
                    cc=self.code[c]
                    l=np.stack([np.bincount(cc[t],weights=lv,minlength=len(self.first[c])) for lv in ll.T],1)+logprior
                    v=logsumexp(l,axis=1)
                    weights.append(v.sum())
                    estimates.append((query_prob*np.exp(l-v[:,None])[cc[q]]).sum(1))
                weights=np.array(weights)
                root_shallow=np.exp(weights-logsumexp(weights))@np.asarray(estimates)
            total=logsumexp(odds,axis=0)
            weights=np.exp(odds-total)[:,code[q]]
            mass[mask]=total-np.log(2.)
            pred[mask]=(weights*np.stack([leaf_prob,*[pred[c] for c in self.child[mask]]])).sum(0)
        return pred[0],root_shallow


def evaluate(y,base,ref,scores,fold,incumbent,masks):
    result,hits={},{}
    for key,p in scores.items():
        assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        vote=p>=.5
        if incumbent and key.startswith("update"):
            vote=np.where(ref!=(base>=.5),ref,vote)
        hit=vote==y
        gain=hit.astype(int)-(ref==y)
        result[key]=dict(accuracy=float(hit.mean()),fixed=int((gain>0).sum()),harmed=int((gain<0).sum()),
                         net_by_fold=[int(gain[fold==k].sum()) for k in np.unique(fold)],
                         corrected={k:int(((gain>0)&m).sum()) for k,m in masks.items()},
                         native_log_loss_before_overrides=float(log_loss(y,p)))
        hits[key]=(hit,gain)
    return result,hits


def run():
    row,y,frame,g,_=load()
    original,target,masks=historical()
    np.testing.assert_array_equal(original,row)
    np.testing.assert_array_equal(target,y)
    partitions={k:Partitions(v) for k,v in partition_axes(frame).items()}
    OUT.mkdir(exist_ok=True)
    plans=[]
    for s in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{s}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            plans.append((str(s),z["fold"],z["base"],z["base"]>=.5))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        for k,v in (("row",row),("y",y)):
            np.testing.assert_array_equal(z[k],v)
        plans.append(("v131",z["fold"],z["v108"],z["v131"]))
    results,history={},{}
    for name,fold,base,ref in plans:
        out={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            pt,qt=teachers(frame,y,g,t,q,1171+int(f))
            ord_train=logit(pt.clip(.002,.998))
            ord_query=logit(base[q].clip(.002,.998))
            fake=y.copy()
            rng=np.random.default_rng(494)
            for band in np.array_split(t[np.argsort(pt)],12):
                fake[band]=rng.permutation(y[band])
            for mode,y_used,a,b,scale in (("native",y,np.zeros(len(t)),np.zeros(len(q)),1.7),
                                         ("update",y,ord_train,ord_query,.8),
                                         ("update_placebo",fake,ord_train,ord_query,.8)):
                preds,shallow=[],[]
                for tree in partitions.values():
                    p,simple=tree.infer(t,q,y_used,a,b,scale)
                    preds.append(p)
                    shallow.append(simple)
                for suffix,bank in (("",preds),("_single_field",shallow)):
                    out.setdefault(mode+suffix,np.full(len(y),np.nan))[q]=np.mean(bank,axis=0)
            out.setdefault("teacher",np.full(len(y),np.nan))[q]=qt.mean(1)
            print(name,f,"complete",flush=True)
        results[name],hits=evaluate(y,base,ref,out,fold,name=="v131",masks)
        if name!="v131":
            for k,v in hits.items():
                history.setdefault(k,[]).append(v)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**out)
        write(REPORT/"results.json",dict(plans=results))
    stable={}
    for k,values in history.items():
        all_hit=np.logical_and.reduce([h for h,_ in values])
        stable[k]={n:int((all_hit&m).sum()) for n,m in masks.items()}
        stable[k]["gain_ci_pp"]=interval(np.mean([d for _,d in values],0),g)
    write(REPORT/"results.json",dict(plans=results,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
