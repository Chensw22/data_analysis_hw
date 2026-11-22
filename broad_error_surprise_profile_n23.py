"""Locate broad field anomalies among false positives and false negatives at comparable risk."""
import numpy as np
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from broad_error_tail_ranking_n23 import ART,REPORT,SEEDS,historical,load,strata,metrics,write


def ranks(values,base,ref):
    # Rank relative atypicality among peers with the same predicted class and risk.
    # These bands do not use any labels. Every passenger belongs to exactly one.
    cells=np.full(len(ref),-1)
    out=np.zeros_like(values,dtype=float)
    for side in (0,1):
        ids=np.flatnonzero(ref==side)
        risk=np.minimum(base[ids],1-base[ids])
        for k,part in enumerate(np.array_split(ids[np.argsort(risk,kind="stable")],20)):
            cells[part]=side*20+k
            out[part]=(rankdata(values[part],axis=0)-.5)/len(part)
    return cells,out


def moments(err,cells,rank):
    n,d=rank.shape
    num=np.zeros((n,2,d))
    den=np.zeros((n,2))
    for c in np.unique(cells):
        ix=cells==c
        v=err[ix].astype(float)
        e=v-v.mean()
        side=c//20
        num[ix,side]=e[:,None]*(rank[ix]-rank[ix].mean(0))
        den[ix,side]=e**2
    return num,den


def contrasts(num,den,g,columns):
    count=den.sum(0)
    delta=num.sum(0)/count[:,None]
    influence=(num-den[:,:,None]*delta)/count[None,:,None]
    group=np.unique(g,return_inverse=True)[1]
    cluster=np.zeros((1+group.max(),2,len(columns)))
    np.add.at(cluster,group,influence)
    se=np.sqrt((cluster**2).sum(0))
    rng=np.random.default_rng(725)
    maxima=[]
    z=cluster.reshape(len(cluster),-1)/se.ravel()
    for _ in range(20):
        # One shared random sign per booking across all estimates and assignments.
        maxima.extend(np.abs(rng.choice([-1.,1.],size=(100,len(cluster)))@z).max(axis=1))
    cut=float(np.quantile(maxima,.95))
    return {str(side):{k:dict(difference_percentiles=float(100*delta[side,j]),
            simultaneous_95=(100*(delta[side,j]+np.array([-1,1])*cut*se[side,j])).tolist())
            for j,k in enumerate(columns)} for side in (0,1)}


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig);np.testing.assert_array_equal(y,labels)
    with np.load(ART/"nov23_manifest_energy/representations.npz") as z:
        np.testing.assert_array_equal(row,z["row"])
        views={k:-z[k][:,24::4].astype(float) for k in ("real","sham")}
    assert views["real"].shape==(len(row),len(frame.columns))
    sums,dens=[],[]
    plans,history={},{}
    scope=dict(**masks,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy(),
                   repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_manifest_energy/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            base,fold,ref=z["base"],z["fold"],z["ref"]
        risk=np.minimum(base,1-base).clip(.001,.4999)
        eligible=ref==(base>=.5)
        err=y!=ref
        cells,actual=ranks(views["real"],base,ref)
        if name!="v131":
            a,b=moments(err,cells,actual)
            sums.append(a);dens.append(b)
        _,sham=ranks(views["sham"],base,ref)
        outcomes={}
        for k,rank in (("risk",np.zeros((len(y),1))),("real",actual),("sham",sham)):
            pred=np.zeros(len(y),bool)
            for f in np.unique(fold):
                ids=np.flatnonzero((fold==f)&eligible)
                for a,b in GroupKFold(3,shuffle=True,random_state=827).split(ids,groups=g[ids]):
                    for side in (False,True):
                        t=ids[a][ref[ids[a]]==side]
                        q=ids[b][ref[ids[b]]==side]
                        assert not np.intersect1d(g[t],g[q]).size
                        x=np.c_[logit(risk),rank]
                        s=StandardScaler().fit(x[t])
                        m=LogisticRegression(C=.06,max_iter=400).fit(s.transform(x[t]),err[t])
                        pred[q]=m.predict_proba(s.transform(x[q]))[:,1]>.5
            outcomes[k]=metrics(pred,ref,y,scope)
            if name!="v131":
                history.setdefault(k,[]).append((ref^pred)==y)
        plans[name]=outcomes
    stats=contrasts(np.mean(sums,axis=0),np.mean(dens,axis=0),g,list(frame.columns))
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in history.items()}
    write(REPORT/"atypical_fields.json",dict(contrasts=stats,plans=plans,stable=stable))


if __name__=="__main__":
    with threadpool_limits(2):
        main()
