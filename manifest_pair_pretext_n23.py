"""Learn which manifest block was exchanged between unrelated travel parties."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write


BILLS=["FoodCourt","ShoppingMall","RoomService","Spa","VRDeck"]
BLOCKS=[BILLS,["Deck","Side","CabinNumber"],["Age","Destination","VIP"]]


def exchange(x,groups,ix,seed):
    clean=x.iloc[ix].reset_index(drop=True)
    frames=[clean]
    target=[np.zeros(len(ix),int)]
    activity=clean[BILLS].gt(0).any(axis=1)
    # Keep the donor broadly plausible: the same observed planet, sleep state and spending status.
    strata=clean.groupby([clean.HomePlanet,clean.CryoSleep,activity],dropna=False).indices
    rng=np.random.default_rng(seed)
    for number,cols in enumerate(BLOCKS,1):
        fake=clean.copy()
        keep=np.zeros(len(ix),bool)
        for members in strata.values():
            if len(members)<2:
                continue
            perm=rng.permutation(members)
            donor=np.roll(perm,len(members)//2)
            permitted=groups[ix[donor]]!=groups[ix[perm]]
            dest,src=perm[permitted],donor[permitted]
            if len(dest):
                old=clean.iloc[dest][cols].reset_index(drop=True)
                new=clean.iloc[src][cols].reset_index(drop=True)
                equal=old.eq(new)|(old.isna()&new.isna())
                keep[dest]=~equal.all(axis=1).to_numpy()
                fake.loc[dest,cols]=new.to_numpy()
        frames.append(fake.loc[keep])
        target.append(np.full(keep.sum(),number))
    return pd.concat(frames,ignore_index=True),np.concatenate(target)


def model(x,y,task):
    return CatBoostClassifier(iterations=160 if task else 500,depth=5,learning_rate=.065,
        auto_class_weights="Balanced" if task else None,thread_count=2,
        random_seed=398,allow_writing_files=False,verbose=False,cat_features=CATEGORICAL).fit(x,y)


def run():
    panel=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    rows=panel.row.to_numpy()
    y,anchors,previous=read_panel()
    np.testing.assert_array_equal(y,panel.target)
    common=~anchors.any(0)
    unresolved=common&~previous.any(axis=(0,1))
    assert common.sum()==1070 and unresolved.sum()==466
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"]).iloc[rows]
    x=compact_features(raw).reset_index(drop=True)
    groups=raw.PassengerId.str.split("_").str[0].to_numpy()
    out=ART/"nov23_pair_pretext"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,rows,y,fold)
        variants={n:np.full(len(y),np.nan) for n in ["direct","integrity","permuted"]}
        pretext=[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(groups[t],groups[q]).size
            # Fit auxiliary scores only on training parties, evaluate genuine and deliberately
            # corrupted validation features without accessing validation outcomes.
            test_frames,test_status=exchange(x,groups,q,839+int(k))
            carrier=np.full((len(t),4),np.nan)
            query_prob=np.zeros((len(q),4))
            diagnostic=np.zeros((len(test_frames),4))
            for j,(a,b) in enumerate(GroupKFold(2,shuffle=True,random_state=819).split(t,groups=groups[t])):
                assert not np.intersect1d(groups[t[a]],groups[t[b]]).size
                synth,synthetic_label=exchange(x,groups,t[a],353+int(k)+j)
                fit=model(synth,synthetic_label,True)
                carrier[b]=fit.predict_proba(x.iloc[t[b]])
                query_prob+=.5*fit.predict_proba(x.iloc[q])
                diagnostic+=.5*fit.predict_proba(test_frames)
            pretext.append(float(balanced_accuracy_score(test_status,diagnostic.argmax(1))))
            assert np.isfinite(carrier).all()
            additions=np.zeros((len(y),4))
            additions[t]=carrier
            additions[q]=query_prob
            shuffled=additions.copy()
            rng=np.random.default_rng(147+int(k))
            shuffled[t],shuffled[q]=additions[rng.permutation(t)],additions[rng.permutation(q)]
            for name,a in [("direct",None),("integrity",additions),("permuted",shuffled)]:
                f=x if a is None else x.join(pd.DataFrame(a,columns=["real","swapped_bill","swapped_cabin","swapped_profile"]))
                variants[name][q]=model(f.iloc[t],y[t],False).predict_proba(f.iloc[q])[:,1]
        record={}
        for name,p in variants.items():
            for desc,score in [(name,p),(name+"_base",.75*base+.25*p)]:
                assert np.isfinite(score).all()
                ok=(score>=.5)==y
                bad=(base>=.5)!=y
                record[desc]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),harmed=int((~bad&~ok).sum()),
                                  hard1070_correct=int(ok[common].sum()),hard466_correct=int(ok[unresolved].sum()))
        results[str(seed)]=dict(results=record,pretext_balanced_accuracy=pretext)
        np.savez_compressed(out/f"{seed}.npz",row=rows,y=y,fold=fold,base=base,p=variants["integrity"],**variants)
        print(seed,[(k,round(v["accuracy"],5),v["hard466_correct"]) for k,v in record.items()],flush=True)
    write(ROOT/"reports/nov23_pair_pretext/result.json",results)


if __name__=="__main__":
    with threadpool_limits(2):
        run()
