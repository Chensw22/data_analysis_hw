"""Train an additive logistic index model by learning directions and shapes jointly."""
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from broad_companion_information_n23 import ART,SEEDS,load,historical,evaluate,calibrated,write,strata
from broad_geographic_transfer_n23 import BILLS,ROOT
from broad_refit_dissent_n23 import features
from projection_shapes_n23 import supervised_directions

OUT=ART/"nov23_learned_indices"
REPORT=ROOT/"reports/nov23_learned_indices"
CENTERS=np.arange(-2,2.1,.8)


def forward(x,linear,axes,coef):
    z=x@axes
    d=(z[:,:,None]-CENTERS)/.8
    h=np.exp(-.5*d*d)
    logit_y=x@linear+(h*coef).sum((1,2))
    grad_z=-(h*d*coef).sum(2)/.8
    return logit_y,h,grad_z


def train(x,y,t,q):
    scale=StandardScaler().fit(x[t])
    a,b=[np.clip(scale.transform(x[ix]),-5,5) for ix in (t,q)]
    axes=supervised_directions(a,y[t])
    axes/=np.maximum((a@axes).std(0),.01)
    a,b=[np.c_[np.ones(len(z)),z] for z in (a,b)]
    axes=np.vstack([np.zeros(axes.shape[1]),axes])
    output={}
    for mode in ("linear","fixed","learned"):
        if mode=="linear":
            output[mode]=calibrated(a,b,y,t)
            continue
        params=[np.zeros(a.shape[1]),axes.copy(),np.zeros((axes.shape[1],len(CENTERS)))]
        m=[np.zeros_like(w) for w in params]
        v=[np.zeros_like(w) for w in params]
        rng=np.random.default_rng(951)
        step=0
        for epoch in range(45):
            for batch in np.array_split(rng.permutation(len(a)),10):
                step+=1
                xb=a[batch]
                linear,dirs,shapes=params
                score,h,grad_z=forward(xb,*params)
                err=(expit(score)-y[t[batch]])/len(batch)
                gradients=[xb.T@err+.001*linear,
                           xb.T@(err[:,None]*grad_z)+.003*(dirs-axes),
                           np.einsum("n,ndk->dk",err,h)+.002*shapes]
                for j,(w,grad) in enumerate(zip(params,gradients)):
                    if mode=="fixed" and j==1:
                        continue
                    m[j]=.9*m[j]+.1*grad
                    v[j]=.999*v[j]+.001*grad**2
                    rate=(.01 if j!=1 else .003)*(1 if epoch<30 else .4)
                    w-=rate*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        output[mode]=expit(forward(b,*params)[0]).clip(1e-7,1-1e-7)
    return output


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig)
    np.testing.assert_array_equal(y,labels)
    spend=frame[BILLS].fillna(0).to_numpy()
    total=spend.sum(1)
    x=np.column_stack([features(frame)[0],np.log1p(total),spend/(1+total[:,None]),spend>0])
    scopes=dict(**masks,repeat_paid=masks["repeat"]&(strata(frame)==2)&y,
                repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy())
    plans,history={},{}
    OUT.mkdir(exist_ok=True)
    for name in [*map(str,SEEDS),"v131"]:
        with np.load(ART/f"nov23_broad_partition_evidence/{name}.npz") as z:
            for k,v in (("row",row),("y",y)):np.testing.assert_array_equal(z[k],v)
            fold,base,ref=z["fold"],z["base"],z["ref"]
        scores={}
        for f in np.unique(fold):
            q=np.flatnonzero(fold==f)
            t=np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            assert not np.intersect1d(g[t],g[q]).size
            for k,p in train(x,y,t,q).items():
                scores.setdefault(k,np.full(len(y),np.nan))[q]=p
        for key in ("fixed","learned"):
            matrix=np.column_stack([logit(v.clip(.001,.999)) for v in (base,scores[key])])
            p=np.zeros(len(y))
            for f in np.unique(fold):
                ids=np.flatnonzero(fold==f)
                for a,b in GroupKFold(3,shuffle=True,random_state=767).split(ids,groups=g[ids]):
                    t,q=ids[a],ids[b]
                    assert not np.intersect1d(g[t],g[q]).size
                    p[q]=calibrated(matrix[t],matrix[q],y,t)
            scores["update_"+key]=p
        result,hits=evaluate(y,base,ref,scores,fold,name=="v131",masks=scopes)
        plans[name]=result
        if name!="v131":
            for k,(hit,_) in hits.items():history.setdefault(k,[]).append(hit)
        np.savez_compressed(OUT/f"{name}.npz",row=row,y=y,fold=fold,base=base,ref=ref,**scores)
        print(name,[(k,round(v["accuracy"],5),v["fixed"],v["harmed"]) for k,v in result.items()],flush=True)
    steady={k:np.logical_and.reduce(h) for k,h in history.items()}
    counts={k:{s:int((hit&m).sum()) for s,m in scopes.items()} for k,hit in steady.items()}
    exclusive={s:int((m&steady["learned"]&~steady["fixed"]&~steady["linear"]).sum()) for s,m in scopes.items()}
    write(REPORT/"results.json",dict(plans=plans,stable=counts,exclusive=exclusive))


if __name__=="__main__":
    with threadpool_limits(1):
        main()
