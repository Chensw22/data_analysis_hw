"""Compare fine passenger-cell outcomes with siblings after adjusting their case mix."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import paired,read_opinions
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write

OUT = ART/'nov23_standardized_refinement'
REPORT = ROOT/'reports/nov23_standardized_refinement'


def code(*cols):
    return pd.factorize(pd.MultiIndex.from_arrays(cols))[0]


def axes(frame):
    age = frame.Age.fillna(-100).to_numpy()
    cabin = frame.CabinNumber.fillna(-1024).to_numpy()
    size = frame.BookingSize.to_numpy()
    planet,dest,deck,side,sleep = [frame[k] for k in ('HomePlanet','Destination','Deck','Side','CryoSleep')]
    mix = code(planet,sleep)
    land = code(deck,sleep)
    return [
        (code(age//20),code(age//10),mix),
        (code(age//10),code(age//5),mix),
        (code(deck,cabin//512),code(deck,cabin//128),mix),
        (code(deck,cabin//128),code(deck,cabin//32),mix),
        (code(deck),code(deck,side),mix),
        (code(planet),code(planet,dest),land),
        (code(dest),code(dest,planet),land),
        (code(size>1),code((size>1),np.minimum(size,5)),mix),
        (code(sleep),code(sleep,frame.VIP),code(planet,deck)),
    ]


def ratio_counts(a,b,c,d):
    # a,b are transported/stayed in child; c,d are outcomes in the rest of its parent.
    # Mantel-Haenszel pooling weights a stratum by available between-group comparisons.
    n = np.maximum(a+b+c+d,1)
    up = (a*d/n).sum(1)
    dn = (b*c/n).sum(1)
    return np.log((up+2)/(dn+2)),np.log1p(up+dn)


def estimate(p,c,z,t,q,y):
    m,n = z.max()+1,c.max()+1
    par = np.zeros(n,dtype=int)
    par[c] = p
    assert (par[c]==p).all()

    def counts(bins,labels,nbin):
        shape = (nbin,m)
        total = np.bincount(bins,minlength=nbin*m).reshape(shape)
        pos = np.bincount(bins,weights=labels,minlength=nbin*m).reshape(shape)
        return pos,total-pos

    def get(mix):
        a,b = counts(c[t]*m+mix,y[t],n)
        ps,ns = counts(p[t]*m+mix,y[t],p.max()+1)
        return a,b,ps[par]-a,ns[par]-b

    true = get(z[t])
    # Each parent keeps every registration stratum's frequency, its total outcome rate,
    # and the exact same fine/sibling counts when the stratum labels are shuffled.
    fake = z[t].copy()
    rng = np.random.default_rng(318)
    for parent in np.unique(p[t]):
        use = np.flatnonzero(p[t]==parent)
        fake[use] = rng.permutation(fake[use])
    marginal = [v.sum(1,keepdims=True) for v in true]
    a,b,s,d = marginal
    prior = (a+s+4)/(a+s+b+d+8)
    crude,precision = ratio_counts(*marginal)
    def correction(tables):
        contrast,support = ratio_counts(*tables)
        opposite = crude*contrast<0
        return np.c_[contrast-crude,support,opposite*(contrast-crude)]
    return [v[c[q]] for v in (np.c_[prior,np.log1p(a+b+s+d)],
        np.c_[crude,precision],correction(get(fake)),correction(true))]


def forecast(frame,cells,y,g,t,q):
    pred = {}
    for a,b in GroupKFold(2,shuffle=True,random_state=713).split(t,groups=g[t]):
        atlas,train = t[a],t[b]
        rows = np.r_[train,q]
        assert not np.intersect1d(g[atlas],g[rows]).size
        raw = frame.copy()
        for key in raw.select_dtypes(exclude=np.number):
            levels = raw[key].iloc[train].unique()
            raw[key] = pd.Categorical(raw[key].where(raw[key].isin(levels)),categories=levels)
        x = pd.get_dummies(raw).to_numpy(float)[rows]
        estimates = [estimate(*cell,atlas,rows,y) for cell in cells]
        coarse,crude,fake,adjust = [np.column_stack(a) for a in zip(*estimates)]
        base = np.c_[x,coarse]
        root = np.c_[base,crude]
        for name,data in dict(raw=x,coarse=base,crude=root,shuffled=np.c_[root,fake],
                              adjusted=np.c_[root,adjust]).items():
            clf = HistGradientBoostingClassifier(max_iter=145,max_leaf_nodes=15,min_samples_leaf=24,
                learning_rate=.07,l2_regularization=5,early_stopping=False,random_state=833)
            clf.fit(data[:len(train)],y[train])
            pred.setdefault(name,np.zeros(len(q)))[:] += .5*clf.predict_proba(data[len(train):])[:,1]
    return pred


def audit_reversals():
    row,y,frame,g,_ = load()
    _,_,hard = cohorts(row,y)
    cells = axes(frame)
    report = {}
    for seed in [*SEEDS,'v131']:
        with np.load(OUT/f'{seed}.npz') as z:
            np.testing.assert_array_equal(row,z['row'])
            fold,ref,adjust,crude = [z[k] for k in ('fold','ref','adjusted','crude')]
        coverage = np.zeros(len(y),bool)
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            splits = []
            for a,_ in GroupKFold(2,shuffle=True,random_state=713).split(t,groups=g[t]):
                signs = []
                for cell in cells:
                    _,rough,_,adjustment = estimate(*cell,t[a],q,y)
                    marginal = rough[:,0]
                    conditional = marginal+adjustment[:,0]
                    reverse = (marginal*conditional<0)&(np.minimum(abs(marginal),abs(conditional))>.1)
                    signs.append(np.sign(conditional)*reverse)
                splits.append(np.column_stack(signs))
            coverage[q] = np.any((splits[0]!=0)&(splits[0]==splits[1]),axis=1)
        result = {}
        for label,mask in [('replicated',coverage),('other',~coverage)]:
            h = (adjust[mask]>=.5)==y[mask]
            result[label] = dict(rows=int(mask.sum()),hard=int((hard&mask).sum()),hard_correct=int(h[hard[mask]].sum()),
                vs_reference=paired(h,ref[mask]==y[mask]),vs_crude=paired(h,(crude[mask]>=.5)==y[mask]))
        report[str(seed)] = result
    write(REPORT/'reversals.json',report)


def main():
    row,y,frame,g,_ = load()
    _,right,hard = cohorts(row,y)
    solo = frame.BookingSize.to_numpy()==1
    cells = axes(frame)
    REPORT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    panel,summary = {},dict(folds={},across={})
    for seed in [*SEEDS,'v131']:
        fold,ref,op,_ = read_opinions(seed,row,y)
        score = {}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for k,v in forecast(frame,cells,y,g,t,q).items():
                score.setdefault(k,np.full(len(y),np.nan))[q] = v
        assert all(np.isfinite(v).all() for v in score.values())
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,**score)
        hits = {'reference':ref==y}
        for k,v in score.items():
            hits[k] = (v>=.5)==y
            hits[k+'_quarter'] = np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*v>=.5,ref)==y
        panel[str(seed)] = hits
        summary['folds'][str(seed)] = {k:dict(**paired(h,ref==y),hard=int(h[hard].sum()),
            solo_hard=int((h&hard&solo).sum()),right_wrong=int((~h&right).sum())) for k,h in hits.items()}
        print(seed,[(k,round(100*v.mean(),3),int(v[hard].sum())) for k,v in hits.items() if k in ('reference','adjusted','adjusted_quarter')],flush=True)
    table = {k:np.stack([panel[str(s)][k] for s in SEEDS]) for k in hits}
    for k,h in table.items():
        comp = 'shuffled_quarter' if k.endswith('_quarter') else 'shuffled'
        summary['across'][k] = dict(hard_all=int((h.all(0)&hard).sum()),hard_any=int((h.any(0)&hard).sum()),
            solo_hard_all=int((h.all(0)&hard&solo).sum()),solo_hard_any=int((h.any(0)&hard&solo).sum()),
            right_wrong_all=int((~h.any(0)&right).sum()),ci_vs_ref=interval((h.astype(float)-table['reference']).mean(0),g),
            ci_vs_shuffled=interval((h.astype(float)-table[comp]).mean(0),g))
    write(REPORT/'results.json',summary)
    audit_reversals()


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
