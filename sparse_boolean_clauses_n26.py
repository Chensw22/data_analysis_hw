"""Learn short signed Boolean conjunctions directly from passenger labels by column generation."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,OUT,data
from price_implied_deck_n26 import FEES


def predicates(raw,groups):
    cabin=raw.Cabin.str.split('/',expand=True)
    columns=[];names=[];fields=[]
    def add(field,name,column):
        columns.append(np.array(column,bool))
        fields.append(field)
        names.append(name)
    for field,series in [('origin',raw.HomePlanet),('sleep',raw.CryoSleep),('dest',raw.Destination),
                         ('vip',raw.VIP),('deck',cabin[0]),('side',cabin[2])]:
        for value in sorted(series.dropna().unique()):
            add(field,f'{field}={value}',series.eq(value))
        add(field,f'{field} missing',series.isna())
    bill=raw[FEES]
    variables=[('age',raw.Age,[8,13,18,25,35,50,65]),
               ('booking size',raw.groupby(groups).PassengerId.transform('size'),[2,3,4]),
               ('cabin number',pd.to_numeric(cabin[1]),[250,500,750,1000,1250,1500]),
               ('total',bill.sum(axis=1,min_count=1),[1,100,500,800,1200,1800,3000]),
               ('services',bill.gt(0).sum(axis=1),[1,2,3])]
    variables.extend((col,bill[col],[1,50,250,500,800,1200,2000]) for col in FEES)
    for field,series,cuts in variables:
        for cut in cuts:
            add(field,f'{field}>={cut}',series>=cut)
        if series.isna().any():
            add(field,f'{field} missing',series.isna())
    x=np.column_stack(columns)
    return np.column_stack([x,~x]),names+['NOT '+n for n in names],np.tile(fields,2)


def activate(x,clauses):
    return np.column_stack([np.all(x[:,list(clause)],axis=1) for clause in clauses]).astype(np.float64)


def model(x,y):
    return LogisticRegression(solver='liblinear',l1_ratio=1.,C=.4,max_iter=700,random_state=619).fit(x,y)


def expand(x,fields,residual,parents,excluded,limit):
    # The inner product with a new clause is the exact negative logistic gradient at coefficient zero.
    # Standardize rankings for support, then fit selected rules jointly using unscaled L1 logistic loss.
    masks=activate(x,parents).astype(np.float32)
    literal=x.astype(np.float32)
    gradient=(masks*residual[:,None]).T.astype(np.float32) @ literal
    count=masks.T @ literal
    ranked=abs(gradient)/np.sqrt(count+25)
    ensure_valid=(count>=18)&(count<=(len(x)-18))
    for i,clause in enumerate(parents):
        ensure_valid[i] &= ~np.isin(fields,fields[list(clause)])
    ranked[~ensure_valid]=-np.inf
    candidates=[]
    for pos in np.argsort(-ranked.ravel(),kind='stable'):
        i,j=divmod(int(pos),x.shape[1])
        if not np.isfinite(ranked[i,j]):
            break
        clause=tuple(sorted((*parents[i],j)))
        if clause not in excluded:
            candidates.append(clause)
            excluded.add(clause)
            if len(candidates)==limit:
                break
    return candidates


def fit(x,fields,y,t,q,names):
    xt=x[t]; xq=x[q]
    singles=[(j,) for j in range(x.shape[1]//2)]
    initial=activate(xt,singles)
    additive=model(initial,y[t])
    control=additive.predict_proba(activate(xq,singles))[:,1]
    residual=y[t]-additive.predict_proba(initial)[:,1]
    clauses=singles.copy()
    excluded=set(clauses)
    base=[(j,) for j in range(x.shape[1])]
    for _ in range(3):
        pairs=expand(xt,fields,residual,base,excluded,56)
        triples=expand(xt,fields,residual,pairs,excluded,56) if pairs else []
        clauses.extend(pairs+triples)
        design=activate(xt,clauses)
        selected=model(design,y[t])
        residual=y[t]-selected.predict_proba(design)[:,1]
    weights=selected.coef_[0]
    top=np.argsort(-abs(weights)*np.array([len(c)>1 for c in clauses]))[:5]
    rules=[dict(condition=[names[k] for k in clauses[i]],coefficient=float(weights[i])) for i in top if weights[i]!=0]
    count=int(sum(len(c)>1 and abs(w)>1e-8 for c,w in zip(clauses,weights,strict=True)))
    strength=abs(weights)*design.std(0)
    active=[i for i,c in enumerate(clauses) if len(c)>1 and abs(weights[i])>1e-8]
    shortlist=sorted(active,key=lambda i:-strength[i])[:12]
    kept=list(range(len(singles)))+shortlist
    compact=model(design[:,kept],y[t])
    query=activate(xq,clauses)
    return control,selected.predict_proba(query)[:,1],compact.predict_proba(query[:,kept])[:,1],count,rules


def run():
    raw,y,g,fold=data()
    x,names,fields=predicates(raw,g)
    single=np.zeros(len(y));joint=single.copy();compact=single.copy()
    counts=[];examples=[]
    for k in np.unique(fold):
        t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        assert not np.isin(g[t],g[q]).any()
        single[q],joint[q],compact[q],num,rules=fit(x,fields,y,t,q,names)
        counts.append(num);examples.append(rules)
        print(int(k),num,flush=True)
    cache=ROOT/'artifacts/nov26_sparse_boolean_clauses'
    cache.mkdir(exist_ok=True)
    archive=cache/'eight_fold_oof.npz'
    arrays=dict(passenger=raw.PassengerId.to_numpy(str),row=np.arange(len(y)),fold=fold,y=y,
                additive=single,full=joint,compact12=compact)
    assert len(np.unique(arrays['passenger']))==len(y)==8693
    assert np.isfinite(np.column_stack([single,joint,compact])).all()
    temporary=archive.with_name('eight_fold_oof.tmp.npz')
    np.savez_compressed(temporary,**arrays)
    temporary.replace(archive)
    with np.load(archive,allow_pickle=False) as z:
        for key,expected in arrays.items():
            np.testing.assert_array_equal(z[key],expected)
    with np.load(OUT/'mitra96.npz') as z:
        for key,v in [('y',y),('fold',fold),('passenger',raw.PassengerId.to_numpy(str))]:
            np.testing.assert_array_equal(z[key],v)
        anchor=z['p'].clip(.003,.997)
    _,cur,_=inputs()
    parent=cur['parent']
    blend=lambda p: (.75*logit(anchor)+.25*logit(p.clip(.003,.997)))>=0
    add,jointblend,smallblend=blend(single),blend(joint),blend(compact)
    actions=dict(twelve=(compact>=.5)!=parent,twelve_blend=smallblend!=parent,
                 twelve_unique=(smallblend!=parent)&(add==parent)&((anchor>=.5)==parent),additive=(single>=.5)!=parent,clauses=(joint>=.5)!=parent,
                 clauses_only=((joint>=.5)!=parent)&((single>=.5)==parent),
                 blend_additive=add!=parent,blend_clauses=jointblend!=parent,
                 blend_unique=(jointblend!=parent)&(add==parent)&((anchor>=.5)==parent))
    report=dict(nonzero_interactions_by_fold=counts,largest_interactions_by_fold=examples,
                eight_fold_accuracy={n:float((p==y).mean()) for n,p in [('additive',single>=.5),('clauses',joint>=.5),
                    ('twelve',compact>=.5),('mitra_twelve',smallblend),('mitra',anchor>=.5),('mitra_additive',add),('mitra_clauses',jointblend)]},scopes={})
    for key,m in [('all',cur['known']),('eleven_agree',cur['common']),('new_orders',cur['known']&cur['new'])]:
        report['scopes'][key]={method:summary(a,parent!=y,m,fold) for method,a in actions.items()}
        report['scopes'][key]['blend_vs_mitra']=summary(jointblend!=(anchor>=.5),(anchor>=.5)!=y,m,fold)
        report['scopes'][key]['blend_vs_additive_blend']=summary(jointblend!=add,add!=y,m,fold)
    report['scope']='Predicates are fixed without labels. Each of eight folds learns AND clauses with two or three distinct fields by logistic-gradient search, with a signed L1-penalized logistic readout. Control uses exactly the same unary inputs. Blend weights fixed before results. New orders are a subset of these same eight held folds.'
    (ROOT/'reports/nov23_matched_learning_curve/sparse_boolean_clauses.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['eight_fold_accuracy'],flush=True)
    print({k:(v['fixed'],v['harmed']) for k,v in report['scopes']['eleven_agree'].items()},flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        run()
