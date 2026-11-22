"""Search every decision table on up to four dissenting model votes; evaluate on other bookings."""
from itertools import combinations
from math import comb
import numpy as np
from scipy.special import betaln,logsumexp
from sklearn.model_selection import GroupKFold
from broad_error_tail_ranking_n23 import ART,REPORT,SEEDS,historical,load,strata,metrics,write

MEMBERS={
    "observed_agreement":[k+"_agreement" for k in ("sleep","earth","side","pay","retail")],
    "learned_indices":["learned"],"interbooking_relations":["relation"],"marker_tree_likelihood":["tree"],
    "broad_boosting_path":["trees_800"],"broad_neural_path":["epoch_80"],
    "broad_basket_balance":["weighted"],"service_assignment":["factor_real"]
}
CUTS=np.array([0,.10,.20,.30,.40,.45,.50001])
MID=(CUTS[1:]+CUTS[:-1])/2


def bank(name,row,y):
    preds,names=[],[]
    for src,keys in MEMBERS.items():
        with np.load(ART/f"nov23_{src}/{name}.npz") as z:
            for k,v in (("row",row),("y",y)): np.testing.assert_array_equal(z[k],v)
            if not preds:
                fold,base,ref=z["fold"],z["base"],z["ref"]
            for k,v in (("fold",fold),("ref",ref)):np.testing.assert_array_equal(z[k],v)
            np.testing.assert_allclose(base.clip(.001,.999),z["base"].clip(.001,.999),atol=1e-10)
            for key in keys:
                preds.append(z[key]>=.5)
                names.append(src+"/"+key)
    return fold,base,ref,np.column_stack(preds)!=ref[:,None],names


def tables(votes,risk,ref):
    base=np.digitize(risk,CUTS[1:-1])+len(MID)*ref
    indices=[];codes=[];priors=[]
    for k in range(5):
        for cols in combinations(range(votes.shape[1]),k):
            bins=2**k
            pattern=votes[:,cols]@(2**np.arange(k)) if k else np.zeros(len(base),int)
            indices.append(cols)
            codes.append((bins*base+pattern).astype(np.int16))
            priors.append(np.repeat(np.tile(MID,2),bins))
    return indices,np.stack(codes),priors


def compare(subsets,codes,priors,t,q,err,nv):
    gains=np.zeros(len(subsets),int)
    evidence=np.zeros(len(subsets))
    predictions=np.zeros((len(subsets),len(q)))
    for j,(cols,code,prior) in enumerate(zip(subsets,codes,priors)):
        ix=code[t]
        n=np.bincount(ix,minlength=len(prior))
        wrong=np.bincount(ix,weights=err[t],minlength=len(prior))
        gains[j]=np.maximum(2*wrong-n,0).sum()
        # Independent beta priors per pattern pool small cells toward the baseline
        # risk band. Uniform prior over model size and over subsets of each size.
        s=18.
        a,b=s*prior,s*(1-prior)
        evidence[j]=np.sum(betaln(a+wrong,b+n-wrong)-betaln(a,b))-np.log(comb(nv,len(cols)))
        predictions[j]=(a+wrong)[code[q]]/(s+n)[code[q]]
    winner=int(np.argmax(gains))
    c=codes[winner];p=priors[winner]
    n=np.bincount(c[t],minlength=len(p))
    wrong=np.bincount(c[t],weights=err[t],minlength=len(p))
    empirical=(2*wrong>n)[c[q]]
    posterior=np.exp(evidence-logsumexp(evidence))
    options=dict(empirical=empirical,shrunk_winner=predictions[winner]>.5,
                 average=(posterior@predictions)>.5,base=predictions[0]>.5)
    return gains,winner,options


def main():
    row,y,frame,g,_=load()
    orig,labels,masks=historical()
    np.testing.assert_array_equal(row,orig);np.testing.assert_array_equal(y,labels)
    scope=dict(**masks,repeat_solo=masks["repeat"]&(frame.BookingSize==1).to_numpy(),
               repeat_paid=masks["repeat"]&(strata(frame)==2)&y)
    plans,history={},{}
    for name in [*map(str,SEEDS),"v131"]:
        fold,base,ref,votes,names=bank(name,row,y)
        eligible=ref==(base>=.5)
        active=np.flatnonzero(eligible)
        err=y!=ref
        subsets,codes,priors=tables(votes,np.minimum(base,1-base),ref)
        best,winner,oracle=compare(subsets,codes,priors,active,active,err,len(names))
        sizes=np.array([len(s) for s in subsets])
        obs={}
        for k in range(5):
            delta=int(best[sizes<=k].max())
            obs[k]=dict(net=delta,accuracy=float(((~err).sum()+delta)/len(y)))
        assert int((oracle["empirical"]*(2*err[active].astype(int)-1)).sum())==obs[4]["net"]
        actions={}
        for f in np.unique(fold):
            ids=np.flatnonzero((fold==f)&eligible)
            for a,b in GroupKFold(3,shuffle=True,random_state=275).split(ids,groups=g[ids]):
                t,q=ids[a],ids[b]
                assert not np.intersect1d(g[t],g[q]).size
                _,_,policies=compare(subsets,codes,priors,t,q,err,len(names))
                for k,v in policies.items():
                    actions.setdefault(k,np.zeros(len(y),bool))[q]=v
        records={k:metrics(act,ref,y,scope) for k,act in actions.items()}
        if name!="v131":
            for k,act in actions.items():history.setdefault(k,[]).append((ref^act)==y)
        plans[name]=dict(tables_searched=len(subsets),upper=obs,retrospective_members=[names[i] for i in subsets[winner]],
                         learned=records)
        print(name,obs[4],records["average"]["accuracy"],flush=True)
    stable={k:{s:int((np.logical_and.reduce(v)&m).sum()) for s,m in scope.items()} for k,v in history.items()}
    write(REPORT/"tables.json",dict(plans=plans,members=names,stable=stable))


if __name__=="__main__":
    main()
