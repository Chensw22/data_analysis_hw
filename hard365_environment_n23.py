"""Compare cross-booking surroundings between matched hard and correct passengers."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from exaone_validation_data import compact_features, CATEGORICAL
from population_error_accounting_n23 import ADDITIONS, BILLS, FAMILIES, ROOT, ART, SEEDS, read_scores, verify
from relational_error_corrector import write


def cohort(rows, y, member):
    hard = ((member >= .5) != y[:, None]).all(axis=1)
    right = np.ones(len(y), bool)
    for seed in SEEDS:
        with np.load(ART / f"nov23_group_kernel/{seed}.npz") as z:
            base, fold = z["base"], z["fold"]
            verify(z, rows, y, fold)
            right &= (base >= .5) == y
        for src, key in [*FAMILIES.values(), ("simplex_likelihood", "budget_prior"), *ADDITIONS.values()]:
            with np.load(ART / f"nov23_{src}/{seed}.npz") as z:
                verify(z, rows, y, fold)
                hard &= (z[key] >= .5) != y
    assert hard.sum() == 365 and right.sum() == 5436
    return hard, right


def build(raw):
    x = compact_features(raw).reset_index(drop=True)
    group = raw.PassengerId.str.split("_").str[0].astype(int)
    size = group.map(group.value_counts()).to_numpy()
    bills = x[BILLS].to_numpy()
    continuous = np.column_stack([x.Age.fillna(29)/20, x.Age.isna(),
                                  np.log1p(np.nan_to_num(bills))/3, np.isnan(bills), np.log1p(size)])
    personal = np.column_stack([pd.get_dummies(x[CATEGORICAL], dtype=float), continuous])
    shares = np.column_stack([x.HomePlanet.eq("Earth"), x.HomePlanet.eq("Europa"), x.CryoSleep.eq("True"),
                              x.Age.lt(14), size == 1, (bills > 0), np.isnan(bills).any(1)]).astype(float)
    # Nearest distinct bookings on each side, weighted equally per booking. A tied
    # berth position is excluded; each direction uses strictly different positions.
    g = group.to_numpy()
    env = []
    side = x.Deck + "_" + x.Side
    for track, position in [(pd.Series("all",index=x.index), g.astype(float)),
                            (side, x.CabinNumber.to_numpy())]:
        block = np.full((len(x), 4, shares.shape[1]+1), np.nan)
        for indices in track.groupby(track).indices.values():
            ids = indices[np.isfinite(position[indices])]
            ids = ids[np.argsort(position[ids],kind="stable")]
            positions = position[ids]
            for i in ids:
                left = np.searchsorted(positions,position[i],side="left")
                right = np.searchsorted(positions,position[i],side="right")
                for d, segment in enumerate([ids[max(0,left-110):left][::-1],ids[right:right+110]]):
                    pool = segment[g[segment]!=g[i]]
                    if not len(pool):
                        continue
                    keys, first = np.unique(g[pool],return_index=True)
                    selected = keys[np.argsort(first)[:12]]
                    if not len(selected):
                        continue
                    # Include all members of a selected booking present on this side of the window.
                    agg = np.array([shares[pool[g[pool]==key]].mean(0) for key in selected])
                    for j,k in enumerate([3,12]):
                        block[i,2*d+j,:-1] = agg[:k].mean(0)
                        block[i,2*d+j,-1] = abs(position[i]-position[pool[g[pool]==selected[min(k,len(selected))-1]][0]])
        block[:,:,-1] = np.log1p(block[:,:,-1])
        env.extend([np.nan_to_num(block,nan=-1).reshape(len(raw),-1)])
    return personal, np.column_stack(env), group.to_numpy(), size, x


def match(x, personal, group, size, y, hard, right):
    hard_ids, controls = np.flatnonzero(hard),np.flatnonzero(right)
    d = cdist(personal[hard_ids],personal[controls],metric="sqeuclidean")
    # Match the prediction direction, so each case-control pair has opposing outcomes.
    d += 1.e7*(y[hard_ids,None] == y[controls])
    d += 1.e7*(group[hard_ids,None] == group[controls])
    states = np.where(x[BILLS].isna(),-1, (x[BILLS]>0).astype(int))
    coarse = np.column_stack([x.HomePlanet,x.CryoSleep,size>1,states]).astype(str)
    for col in coarse.T:
        d += 25 * (col[hard_ids,None]!=col[controls])
    a,b = linear_sum_assignment(d)
    cases,controls=hard_ids[a],controls[b]
    assert np.all(y[cases]!=y[controls]) and np.all(group[cases]!=group[controls]) and len(cases)==365
    exact=(coarse[cases]==coarse[controls]).all(1)
    return cases,controls,exact


def adjust(z, t, q, base, y):
    # Fixed strong shrinkage, with dimensions all on comparable scales.
    std=z[t].std(0).clip(.25)
    design=((z-z[t].mean(0))/std).clip(-5,5)
    train=design[t]
    prior=logit(base.clip(.001,.999))
    def loss(w):
        f=prior[t]+train@w
        return np.sum(np.logaddexp(0,f)-y[t]*f)+150*np.square(w).sum(), train.T@(expit(f)-y[t])+300*w
    fit=minimize(loss,np.zeros(z.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":200})
    assert fit.success
    return expit(prior[q]+design[q]@fit.x)


def run():
    rows,y,p=read_scores()
    hard,right=cohort(rows,y,p)
    raw=pd.concat([pd.read_csv(ROOT/f"data/raw/{part}.csv",usecols=lambda c:c not in {"Name","Transported"})
                   for part in ["train","test"]],ignore_index=True)
    personal,env,groups,size,x=build(raw)
    x=x.iloc[rows].reset_index(drop=True)
    personal,env,groups,size=[a[rows] for a in (personal,env,groups,size)]
    cases,ctrl,exact=match(x,personal,groups,size,y,hard,right)
    result={"hard":int(hard.sum()),"exact_pairs":int(exact.sum()),"seeds":{}}
    folder=ART/"nov23_hard365_environment"
    folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        rng=np.random.default_rng(372)
        shuffled=env[rng.permutation(len(env))]
        pred={}
        own=np.column_stack([np.ones(len(y)),base,personal])
        for mode,features in [("own",own),("own_env",np.column_stack([own,env,env*(base[:,None]-.5)])),
                              ("shuffled",np.column_stack([own,shuffled,shuffled*(base[:,None]-.5)]))]:
            score=np.full(len(y),np.nan)
            for k in np.unique(fold):
                held=np.flatnonzero(fold==k)
                for a,b in GroupKFold(3,shuffle=True,random_state=623).split(held,groups=groups[held]):
                    t,q=held[a],held[b]
                    assert not np.intersect1d(groups[t],groups[q]).size
                    score[q]=adjust(features,t,q,base,y)
            assert np.isfinite(score).all()
            pred[mode]=score
        m={}
        for key,score in pred.items():
            hit=(score>=.5)==y
            bh=(base>=.5)==y
            extra=(logit(score.clip(1e-7,1-1e-7))-logit(pred["own"].clip(1e-7,1-1e-7)))
            diff=(extra[cases]-extra[ctrl])*np.where(y[cases],1,-1)
            m[key]=dict(accuracy=float(hit.mean()),fixed=int((hit&~bh).sum()),harmed=int((~hit&bh).sum()),
                        hard_hits=int(hit[hard].sum()),control_harm=int((~hit[ctrl]).sum()),
                        correct_extra_order_exact=float(((diff[exact]>0)+.5*(diff[exact]==0)).mean()),
                        correct_extra_order_all=float(((diff>0)+.5*(diff==0)).mean()))
        result["seeds"][seed]=m
        np.savez_compressed(folder/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,cases=cases,controls=ctrl,exact=exact,**pred)
        print(seed,m,flush=True)
    write(ROOT/"reports/nov23_hard365_environment/result.json",result)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
