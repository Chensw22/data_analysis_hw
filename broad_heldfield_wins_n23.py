"""Compare positive and negative donors on attributes withheld from donor selection."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import BILLS,load
from broad_local_profile_fit_n23 import cohorts
from broad_pairwise_adjudication_n23 import read_opinions,paired
from broad_partition_evidence_n23 import ART,ROOT,SEEDS,write

BLOCKS = [['HomePlanet','Destination'],['Age','CryoSleep','VIP'],
          ['Deck','Side','CabinNumber'],['BookingSize'],['Spa','RoomService','VRDeck'],['FoodCourt','ShoppingMall']]
ARMS = ['raw','local_rate','shuffled','field_wins']
REPORT = ROOT/'reports/nov23_heldfield_wins'
OUT = ART/'nov23_heldfield_wins'


def design(frame,atlas):
    fields = {}
    for names in BLOCKS:
        for name in names:
            col = frame[name]
            if pd.api.types.is_numeric_dtype(col):
                values = col.to_numpy(float)
                observed = np.isfinite(values)
                reference = np.sort(values[atlas][observed[atlas]])
                rank = (np.searchsorted(reference,values,'left')+np.searchsorted(reference,values,'right'))/(2*len(reference))
                fields[name] = np.c_[np.where(observed,rank,0),~observed]
            else:
                fields[name] = np.column_stack([(col==cat).to_numpy(float) for cat in col.iloc[atlas].unique()])/np.sqrt(2)
    return fields


def win(ranks,labels):
    # Descending U means a randomly drawn positive is closer than a negative.
    pos = labels.sum(1)
    neg = labels.shape[1]-pos
    losses = (ranks*labels).sum(1)-pos*(pos+1)/2
    return (.5*pos*neg-losses)/(128+pos*neg)


def features(frame,groups,atlas,query,y):
    assert len(np.intersect1d(groups[atlas],groups[query]))==0
    fields = design(frame,atlas)
    atlas = np.random.default_rng(182).permutation(atlas)
    raw = np.column_stack(list(fields.values()))[query]
    rates, real, sham = [],[],[]
    permutation = np.random.default_rng(642).permutation(48)
    for names in BLOCKS:
        # The attributes that decide who is similar exclude the entire block being judged.
        other = np.column_stack([v for k,v in fields.items() if k not in names])
        subgroup = []
        for start in range(0,len(query),192):
            batch = query[start:start+192]
            d = cdist(other[batch],other[atlas],'sqeuclidean')
            closest = np.argpartition(d,47,axis=1)[:,:48]
            # Randomized atlas order breaks ties independently of outcome.
            subgroup.append(atlas[closest])
        ids = np.concatenate(subgroup)
        labels = y[ids].astype(float)
        rates.append(labels.mean(1))
        distances = [((fields[name][query,None]-fields[name][ids])**2).sum(2) for name in names]
        if len(names)>1:
            distances.append(np.sum(distances,axis=0))
        for gap in distances:
            ranks = rankdata(gap,axis=1)
            real.append(win(ranks,labels))
            sham.append(win(ranks,labels[:,permutation]))
    rate = np.column_stack([raw,*rates])
    return dict(raw=raw,local_rate=rate,shuffled=np.column_stack([rate,*sham]),
                field_wins=np.column_stack([rate,*real]))


def predict(frame,g,t,q,y):
    pred = {key:np.zeros(len(q)) for key in ARMS}
    for a,b in GroupKFold(2,shuffle=True,random_state=448).split(t,groups=g[t]):
        atlas,cal = t[a],t[b]
        for key,x in features(frame,g,atlas,np.r_[cal,q],y).items():
            model = HistGradientBoostingClassifier(max_iter=155,max_leaf_nodes=15,min_samples_leaf=23,
                learning_rate=.07,l2_regularization=4.,early_stopping=False,random_state=114)
            model.fit(x[:len(cal)],y[cal])
            pred[key] += .5*model.predict_proba(x[len(cal):])[:,1]
    return pred


def main():
    row,y,frame,g,_ = load()
    _,right,hard = cohorts(row,y)
    solo = frame.BookingSize.to_numpy()==1
    OUT.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    panel,summary = {},dict(folds={},across={})
    for seed in [*SEEDS,'v131']:
        fold,ref,op,_ = read_opinions(seed,row,y)
        scores = {key:np.full(len(y),np.nan) for key in ARMS}
        for f in np.unique(fold):
            q = np.flatnonzero(fold==f)
            t = np.flatnonzero((fold!=f)&~np.isin(g,g[q]))
            for key,p in predict(frame,g,t,q,y).items():
                scores[key][q] = p
        assert all(np.isfinite(v).all() for v in scores.values())
        np.savez_compressed(OUT/f'{seed}.npz',row=row,y=y,fold=fold,ref=ref,**scores)
        hits = {'reference':ref==y}
        for key,p in scores.items():
            hits[key] = (p>=.5)==y
            hits[key+'_quarter'] = np.where(ref==(op[:,0]>=.5),.75*op[:,0]+.25*p>=.5,ref)==y
        summary['folds'][str(seed)] = {
            key:dict(**paired(h,ref==y),hard=int((h&hard).sum()),solo_hard=int((h&hard&solo).sum()),
                     right_missed=int((~h&right).sum())) for key,h in hits.items()}
        panel[str(seed)] = hits
        print(seed,[(key,round(100*np.mean(v),3),int(v[hard].sum())) for key,v in hits.items()],flush=True)
    cells = {key:np.stack([panel[str(s)][key] for s in SEEDS]) for key in hits}
    for key,h in cells.items():
        control = 'shuffled_quarter' if key.endswith('_quarter') else 'shuffled'
        summary['across'][key] = dict(hard_all=int((h.all(0)&hard).sum()),hard_any=int((h.any(0)&hard).sum()),
            solo_hard_all=int((h.all(0)&hard&solo).sum()),solo_hard_any=int((h.any(0)&hard&solo).sum()),
            right_wrong_all=int((~h.any(0)&right).sum()),ci_vs_ref=interval((h.astype(float)-cells['reference']).mean(0),g),
            ci_vs_shuffled=interval((h.astype(float)-cells[control]).mean(0),g))
    write(REPORT/'results.json',summary)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
