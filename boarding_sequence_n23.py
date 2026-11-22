"""Compare real and permuted local booking sequences across unseen bookings."""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from error_overlap_diagnostic import load_authenticated
from exaone_validation_data import CATEGORICAL
from exact_fold_error_selector import counts
from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
SEEDS=[20261112,20261115,20261116]


def sequence_features(frame, party):
    """Treat a booking only once per deck/side and exclude the whole focal booking."""
    features={}
    values=pd.DataFrame({"deck":frame.Deck,"side":frame.Side,"party":party,"berth":frame.CabinNumber})
    for _, part in values.dropna(subset=["berth"]).groupby(["deck","side"]):
        knots=part.groupby("party",sort=True).berth.median()
        g=knots.index.to_numpy(float)
        b=knots.to_numpy()
        pos=np.searchsorted(g,part.party)
        for width in [1,16]:
            # No extrapolation at the ends; missing context is visible to the model.
            ok=(pos>=width)&(pos+width<len(g))
            if not ok.any():
                continue
            i=pos[ok]
            a,c=i-width,i+width
            frac=(g[i]-g[a])/(g[c]-g[a])
            expected=b[a]+frac*(b[c]-b[a])
            v={"offset":part.berth.to_numpy()[ok]-expected,
               "booking_span":np.log1p(g[c]-g[a]),"berth_span":np.log1p(np.abs(b[c]-b[a])),
               "booking_position":frac}
            for name,vector in v.items():
                key=f"sequence_{width}_{name}"
                features.setdefault(key,np.full(len(frame),np.nan))[part.index.to_numpy()[ok]]=vector
    return pd.DataFrame(features,index=frame.index)


def hard_rows(dev,y):
    records=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    np.testing.assert_array_equal(records.row,dev)
    np.testing.assert_array_equal(records.target,y)
    mask=records.v131_wrong.to_numpy() & records.number_new_correct.eq(0).to_numpy()
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            np.testing.assert_array_equal(ref["row"],dev)
            np.testing.assert_array_equal(ref["y"],y)
            mask &= np.column_stack([(ref[k]>=.5)!=y for k in ["base","fixed","learned_twice"]]).all(1)
    assert mask.sum()==447
    return mask


def run():
    inp,_,_,_,_=load_authenticated()
    dev,y=inp["dev"],inp["y"].astype(bool)
    frame=inp["features"].reset_index(drop=True)
    tickets=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId"]).PassengerId
                       for s in ["train","test"]],ignore_index=True)
    party=tickets.str.split("_").str[0].astype(int)
    orig=frame.assign(booking_order=party,booking_size=party.map(party.value_counts()))
    sequence=sequence_features(frame,party)
    placebo=sequence.copy()
    rng=np.random.default_rng(1123)
    blocks=frame.groupby([frame.Deck,frame.Side,pd.qcut(party,5,labels=False)]).indices
    for indices in blocks.values():
        placebo.iloc[indices]=sequence.iloc[rng.permutation(indices)].to_numpy()
    variants={"basic":orig,"sequence":orig.join(sequence),"placebo":orig.join(placebo)}
    hard=hard_rows(dev,y)
    results={}
    out=ART/"nov23_boarding_sequence"
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as data:
            np.testing.assert_array_equal(data["row"],dev)
            np.testing.assert_array_equal(data["y"],y)
            base,fold=data["base"],data["fold"]
        probabilities={}
        for key,features in variants.items():
            p=np.full(len(dev),np.nan)
            for k in np.unique(fold):
                val=fold==k
                assert not np.intersect1d(party.to_numpy()[dev[val]],party.to_numpy()[dev[~val]]).size
                model=CatBoostClassifier(iterations=520,depth=5,learning_rate=.05,verbose=False,
                                         random_seed=311,thread_count=3,allow_writing_files=False)
                model.fit(features.iloc[dev[~val]],y[~val],cat_features=CATEGORICAL)
                p[val]=model.predict_proba(features.iloc[dev[val]])[:,1]
            assert np.isfinite(p).all()
            probabilities[key]=p
        summary={}
        for key,p in probabilities.items():
            for name,score in [(key,p),(key+"_blend",(base*3+p)/4)]:
                m=counts(score>=.5,base>=.5,y,fold,hard,hard)
                summary[name]={s:m[s] for s in ["accuracy","fixed","harmed","fold_net"]}
                summary[name]["previous_447_correct"]=int(((score[hard]>=.5)==y[hard]).sum())
        results[str(seed)]=summary
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**probabilities)
        print(seed,{k:(round(v["accuracy"],5),v["previous_447_correct"]) for k,v in summary.items()},flush=True)
    write(ROOT/"reports/nov23_boarding_sequence/result.json",dict(results=results,
          coverage={k:int(v.notna().iloc[dev].sum()) for k,v in sequence.items()},
          note="447 mask used only after predictions. Permutations preserve deck, side and booking-order quintile."))


if __name__=="__main__":
    run()
