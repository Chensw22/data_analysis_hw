"""Hold the query's original score fixed and progressively reveal labels of unrelated support bookings."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from broad_booking_kriging_transfer_n23 import ART,ROOT,SEEDS,REPORT,load,summary,geometry,smooth,teachers,cohorts
from relational_error_corrector import write


def run():
    row,y,frame,g,_=load()
    masks=cohorts(row,y)
    geo=geometry()
    plans=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(row,z["row"])
            np.testing.assert_array_equal(y,z["y"])
            plans.append((str(seed),z["fold"],z["base"],None))
    with np.load(ART/"nov23_incumbent_budget_transfer/oof.npz") as z:
        hist=pd.read_csv(ART/"nov21_case_checks/v136_final_oof.csv")
        for key,v in (("row",row),("true",y),("fold",z["fold"]),("v131",z["v131"])):
            np.testing.assert_array_equal(hist[key],v)
        plans.append(("v131",z["fold"],z["v108"],z["v131"]))
    report={}
    hits={}
    out=ART/"nov23_broad_geographic_transfer"
    for name,fold,base,inc in plans:
        bank={}
        samples=[]
        for k in np.unique(fold):
            outer=np.flatnonzero(fold==k)
            train=np.flatnonzero((fold!=k)&~np.isin(g,g[outer]))
            seed=20261112+int(k) if inc is not None else int(name)+int(k)
            p,query_members=teachers(frame,y,g,train,outer,seed)
            cal=LogisticRegression(C=1).fit(logit(p.clip(.002,.998))[:,None],y[train])
            prior=np.full(len(y),np.nan)
            prior[train]=cal.predict_proba(logit(p.clip(.002,.998))[:,None])[:,1]
            # One inner teacher per added booking: identical training support
            # size to a residual from the original training fold.
            keys,codes=np.unique(g[outer],return_inverse=True)
            member=np.random.default_rng(983).permutation(len(keys))[codes]%query_members.shape[1]
            hold=query_members[np.arange(len(outer)),member]
            prior[outer]=cal.predict_proba(logit(hold.clip(.002,.998))[:,None])[:,1]
            for a,b in GroupKFold(4,shuffle=True,random_state=941).split(outer,groups=g[outer]):
                extra,q=outer[a],outer[b]
                t=np.r_[train,extra]
                assert not np.intersect1d(g[t],g[q]).size and np.isfinite(prior[t]).all()
                known,idx=np.unique(g[t],return_inverse=True)
                count=np.bincount(idx)
                residual=np.bincount(idx,weights=y[t].astype(float)-prior[t])/count
                avg=np.bincount(idx,weights=prior[t])/count
                query,back=np.unique(g[q],return_inverse=True)
                all_groups=np.r_[known,query]
                qr=np.arange(len(known),len(all_groups))
                delta,null,_=smooth(all_groups,geo,np.arange(len(known)),qr,residual,avg)
                samples.append([len(train),len(t),len(known),len(q)])
                for key,d in (("dense",delta),("matched_null",null)):
                    bank.setdefault(key,np.full(len(y),np.nan))[q]=(base[q]+d[back]).clip(.001,.999)
            print(name,"density fold",int(k),flush=True)
        if inc is not None:
            changed=inc!=(base>=.5)
            for label in bank:
                bank[label]=np.where(changed,inc.astype(float),bank[label])
            ref=inc.astype(float)
        else:
            ref=base
            for label,pred in bank.items():
                hits.setdefault(label,[]).append((pred>=.5)==y)
        with np.load(out/f"all_training_{name}.npz") as old:
            for key,val in (("row",row),("fold",fold),("y",y),("base",ref)):
                np.testing.assert_array_equal(old[key],val)
            sparse=old["full"]
        report[name]=dict(vs_reference=summary(y,bank,ref,fold,masks),
                          vs_sparse=summary(y,bank,sparse,fold,masks),sample_sizes=samples)
        np.savez_compressed(out/f"dense_{name}.npz",row=row,fold=fold,y=y,base=ref,**bank)
        write(REPORT/"density.json",dict(plans=report))
    stable={k:{n:int((np.logical_and.reduce(v)&m).sum()) for n,m in masks.items()} for k,v in hits.items()}
    write(REPORT/"density.json",dict(plans=report,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
