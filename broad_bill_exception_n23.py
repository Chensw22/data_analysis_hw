"""Train against the marginal association between any bill and the transport label."""
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import BILLS, load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, evaluate, historical, write
from broad_refit_dissent_n23 import features
from broad_training_membership_n23 import model

REPORT = ROOT / "reports/nov23_broad_bill_exception"


def strata(frame):
    bills = frame[BILLS]
    # Incomplete bills with a recorded positive amount are still known spenders.
    state = np.where(bills.gt(0).any(axis=1), 2, bills.isna().any(axis=1).astype(int))
    return state


def neutral_weights(y, state):
    rates = np.array([y[state==s].mean() for s in range(3)])
    p = np.where(y, rates[state], 1-rates[state])
    weights = .5/p
    for s in range(3):
        ix = state==s
        assert np.isclose(weights[ix].sum(),ix.sum()) and np.isclose(weights[ix]@y[ix],ix.sum()/2)
    return weights, rates


def run():
    rows,y,frame,g,_ = load()
    orig,labels,masks = historical()
    np.testing.assert_array_equal(orig,rows)
    np.testing.assert_array_equal(labels,y)
    state = strata(frame)
    x = features(frame)[0]
    exception = ((state==2)&y)|((state==0)&~y)
    scopes = dict(**masks,positive_transported=(state==2)&y,zeros_not_transported=(state==0)&~y,
                  repeat_spent=masks["repeat"]&(state==2)&y,repeat_zeros=masks["repeat"]&(state==0)&~y)
    assert ((state==0)|(state==1)|(state==2)).all()
    records, histories = {},{}
    out = ART / "nov23_broad_bill_exception"
    out.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for k,v in (("row",rows),("y",y)):
                np.testing.assert_array_equal(z[k],v)
            fold,base,ref = z["fold"],z["base"],z["ref"]
        scores = {k:np.full(len(y),np.nan) for k in ("plain","neutral","restored","shuffled")}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            weights,rates = neutral_weights(y[t],state[t])
            sham = weights.copy()
            rng = np.random.default_rng(561)
            for s in range(3):
                select = np.flatnonzero(state[t]==s)
                sham[select] = rng.permutation(weights[select])
            for k,w in (("plain",np.ones(len(t))),("neutral",weights),("shuffled",sham)):
                m = model(240).fit(x[t],y[t],sample_weight=w)
                scores[k][q] = m.predict_proba(x[q])[:,1]
            scores["restored"][q] = expit(logit(scores["neutral"][q])+logit(rates[state[q]]))
        for key in ("plain","neutral","restored","shuffled"):
            # Give both real and sham scores identical access to baseline and bill state.
            z=np.column_stack([logit(base.clip(.001,.999)),logit(scores["plain"]),
                               logit(scores[key]),state==0,state==1])
            z=np.c_[z,z[:,2,None]*(state[:,None]==np.array([0,1]))]
            p=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=436).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(z[t],z[q],y,t)
            scores["update_"+key]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",scopes)
        for k,(hit,gain) in hits.items():
            result[k]["paired_in_exception"]=dict(fixed=int(((gain>0)&exception).sum()),
                                                   harmed=int(((gain<0)&exception).sum()))
            if name!="v131":
                histories.setdefault(k,[]).append(hit)
        records[name]=result
        np.savez_compressed(out/f"{name}.npz",row=rows,y=y,base=base,fold=fold,ref=ref,**scores)
    stable={k:{s:int((np.logical_and.reduce(vals)&mask).sum()) for s,mask in scopes.items()}
            for k,vals in histories.items()}
    write(REPORT/"results.json",dict(plans=records,stable=stable,
          sizes={s:int(mask.sum()) for s,mask in scopes.items()}))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
