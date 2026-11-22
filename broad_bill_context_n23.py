"""Can independent booking context separate billing exceptions from costly false alarms?"""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import ART, ROOT, SEEDS, historical, load, strata, evaluate, write, model
from broad_matched_companions_n23 import data
from broad_refit_dissent_n23 import features

REPORT = ROOT/"reports/nov23_broad_bill_exception"
OUT = ART/"nov23_broad_bill_context"


def expert(x,y,t,q,partition):
    p=np.full(len(q),np.nan)
    for s in np.unique(partition[q]):
        src=t[partition[t]==s]
        take=partition[q]==s
        p[take]=model(240).fit(x[src],y[src]).predict_proba(x[q[take]])[:,1]
    assert np.isfinite(p).all()
    return p


def run():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    state=strata(frame)
    other,peer_groups,size,peer,_=data(row)
    np.testing.assert_array_equal(g,peer_groups.astype(int))
    np.testing.assert_array_equal(size,frame.BookingSize)
    assert other.equals(frame.drop(columns="BookingSize"))
    x=np.column_stack([features(frame)[0],peer])
    scopes=dict(**masks,repeat_spent=masks["repeat"]&(state==2)&y,
                 repeat_zeros=masks["repeat"]&(state==0)&~y)
    records, histories={},{}
    OUT.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_bill_exception/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            fold,base,ref,neutral=z["fold"],z["base"],z["ref"],z["neutral"]
        scores={k:np.zeros(len(y)) for k in ("pooled","conditional","sham")}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            fake=state.copy()
            rng=np.random.default_rng(538)
            for ix in (t,q):
                fake[ix]=rng.permutation(state[ix])
            for key,partition in (("pooled",np.zeros(len(y),int)),("conditional",state),("sham",fake)):
                scores[key][q]=expert(x,y,t,q,partition)
        original=base>=.5
        eligible=((neutral>=.5)!=original)&(original==ref)
        eligible_counts={}
        for s in (0,1,2):
            sel=eligible&(state==s)
            eligible_counts[str(s)]=dict(total=int(sel.sum()),right_to_change=int((sel&(ref!=y)).sum()))
        diagnostics={}
        for key in ("pooled","conditional","sham"):
            # New booking evidence ranks the *actual* disagreements with the neutral specialist.
            v=np.column_stack([logit(np.clip(p,.001,.999)) for p in (base,neutral,scores[key])])
            # Model P(strong baseline is wrong), orienting scores towards the alternative.
            signed=np.where(original[:,None],-v,v)
            ds=np.column_stack([signed,*[signed*(state==s)[:,None] for s in (0,1)],state==0,state==1,original])
            pred=original.copy()
            rank=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&eligible)
                for a,b in GroupKFold(3,shuffle=True,random_state=569).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    scale=StandardScaler().fit(ds[t])
                    lr=LogisticRegression(C=.04,max_iter=400).fit(scale.transform(ds[t]),ref[t]!=y[t])
                    rank[q]=lr.predict_proba(scale.transform(ds[q]))[:,1]
                    pred[q]^=rank[q]>.5
            scores["update_"+key]=np.where(pred,.75,.25)
            diagnostics[key]=float(roc_auc_score(ref[eligible]!=y[eligible],rank[eligible]))
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        records[name]=dict(models=result,disagreements=eligible_counts,ranking=diagnostics)
        for key,(hit,_) in hits.items():
            if name!="v131":
                histories.setdefault(key,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,base=base,ref=ref,fold=fold,**scores)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scopes.items()} for k,v in histories.items()}
    write(REPORT/"context.json",dict(plans=records,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
