"""Learn field-block correction sizes using unseen calibration bookings and test them on a third set."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_conditional_raw_signal_n23 import ART,ROOT,SEEDS,ACTIONS,REPORT,summarize,innovation,predictions
from conditional_field_value_n23 import field_blocks
from peer_attribute_surprise_n23 import peer_means
from relational_error_corrector import write

NAMES=("bills","registration","cabin","field_interactions","peers")


def blocks(raw,row):
    fields,group=field_blocks(raw,np.arange(len(raw)))
    def clean(x):
        x=np.asarray(x,dtype=float)
        return np.column_stack([np.nan_to_num(x,nan=0),np.isnan(x).astype(float)])
    bill=clean(np.column_stack([fields["bill_amounts"],fields["bill_states"]]))
    registry=clean(np.column_stack([fields["origin_destination"],fields["age_sleep_vip"]]))
    cabin=clean(fields["cabin_booking"])
    own=np.column_stack([bill,registry,cabin])
    peers=peer_means(own,group)
    return [v[row] for v in (bill,registry,cabin,own,peers)],group[row]


def expansion(parts,train):
    bank=[]
    columns=[]
    rng=np.random.default_rng(399)
    for n,x in enumerate(parts):
        scale=x[train].std(0).clip(.15)
        v=((x-x[train].mean(0))/scale).clip(-5,5)
        omega=rng.normal(size=(v.shape[1],24))/np.sqrt(v.shape[1])
        wave=np.cos(v@omega+rng.uniform(0,2*np.pi,24))
        block=wave if n==3 else np.column_stack([v,wave])
        bank.append(block)
        columns.extend([n]*block.shape[1])
    return np.column_stack(bank),np.array(columns)


def train_corrections(scores,x,columns,y,t,q):
    base=LogisticRegression(C=.08,max_iter=400).fit(scores[t],y[t])
    offset=base.decision_function(scores[t])
    query=base.decision_function(scores[q])
    ra,rb,_=innovation(scores,x,t,q)
    delta=[]
    for n in range(len(NAMES)):
        a=ra[:,columns==n]
        b=rb[:,columns==n]
        def objective(w):
            logits=offset+a@w
            return np.logaddexp(0,logits).sum()-y[t]@logits+6*(w@w), a.T@(expit(logits)-y[t])+12*w
        opt=minimize(objective,np.zeros(a.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":300})
        assert opt.success,opt.message
        delta.append(b@opt.x)
    return query,np.column_stack(delta)


def weights(offset,delta,y):
    def objective(w):
        logits=offset+delta@w
        return np.logaddexp(0,logits).sum()-y@logits+2*(w@w), delta.T@(expit(logits)-y)+4*w
    opt=minimize(objective,np.zeros(delta.shape[1]),jac=True,method="L-BFGS-B",bounds=[(0,1)]*delta.shape[1])
    assert opt.success,opt.message
    return opt.x


def evaluate(scores,parts,y,group,fold):
    pred={}
    mixing=[]
    for f in np.unique(fold):
        train,outer=np.flatnonzero(fold!=f),np.flatnonzero(fold==f)
        train=train[~np.isin(group[train],group[outer])]
        x,cols=expansion(parts,train)
        for a,b in GroupKFold(3,shuffle=True,random_state=387).split(outer,groups=group[outer]):
            learn,test=outer[a],outer[b]
            assert not np.intersect1d(group[learn],group[test]).size
            cross=np.full((len(learn),1+len(NAMES)),np.nan)
            query=[]
            for ia,ib in GroupKFold(3,shuffle=True,random_state=945).split(learn,groups=group[learn]):
                t,v=learn[ia],learn[ib]
                assert not np.intersect1d(group[t],group[np.r_[v,test]]).size
                o,d=train_corrections(scores,x,cols,y,t,np.r_[v,test])
                cross[ib]=np.column_stack([o[:len(v)],d[:len(v)]])
                query.append((o[len(v):],d[len(v):]))
            assert np.isfinite(cross).all()
            w=weights(cross[:,0],cross[:,1:],y[learn])
            mixing.append(w.tolist())
            settings={"scores":np.zeros(len(w)),"unshrunk":np.ones(len(w)),"honest":w}
            for name,v in settings.items():
                p=np.mean([expit(o+d@v) for o,d in query],axis=0)
                pred.setdefault(name,np.full(len(y),np.nan))[test]=p
    assert all(np.isfinite(p).all() for p in pred.values())
    return pred,mixing


def run():
    inv=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row,y=inv.row.to_numpy(),inv.target.to_numpy(bool)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=lambda c:c not in ("Name","Transported"))
                   for s in ("train","test")],ignore_index=True)
    parts,group=blocks(raw,row)
    result={}
    refs=[]
    mixing={}
    out=ART/"nov23_broad_honest_field_weights"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        base,members,fold=predictions(row,y,seed,ACTIONS)
        refs.append(base)
        src={"registration":members[:,[list(ACTIONS).index("registration_only")]],"baseline":base[:,None],
             "full_panel":np.column_stack([base,members])}
        saved={}
        for name,source in src.items():
            pred,ws=evaluate(logit(source.clip(.002,.998)),parts,y,group,fold)
            mixing.setdefault(name,[]).extend(ws)
            for k,p in pred.items():
                result.setdefault(name,{}).setdefault(k,[]).append(p)
                saved[name+"_"+k]=p
            print(seed,name,round(float(((pred["honest"]>=.5)==y).mean()),4),flush=True)
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,**saved)
    write(REPORT/"nested_weights.json",dict(scores=summarize(result,np.stack(refs),row,y,group),blocks=NAMES,weights=mixing))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
