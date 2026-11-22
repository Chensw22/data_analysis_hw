"""Evaluate frozen English word embeddings on de-identified manifest attributes."""
import json
import platform
import subprocess

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART,ROOT,SEEDS
from exaone_validation_data import compact_features,CATEGORICAL
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

BILLS=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
MEANINGS={
    "HomePlanet":{"Earth":"earth planet","Europa":"europa moon","Mars":"mars planet"},
    "Destination":{"TRAPPIST-1e":"star trappist planet","55 Cancri e":"star cancer planet",
                   "PSO J318.5-22":"star isolated planet"},
    "VIP":{"True":"priority luxury passenger","False":"ordinary passenger"},
    "CryoSleep":{"True":"cryogenic frozen asleep","False":"awake active"},
    "Side":{"P":"port left","S":"starboard right"},
    "Deck":{v:"deck "+v.lower() for v in "ABCDEFGT"}}
SERVICES=["room service lodging","restaurant food court","shopping mall retail","spa wellness","virtual reality entertainment"]
SWIFT='''import NaturalLanguage
import Foundation
guard let m = NLEmbedding.wordEmbedding(for: .english) else { exit(2) }
let words = Array(CommandLine.arguments.dropFirst())
var v = [String: [Double]]()
for word in words { if let vec = m.vector(for:word) { v[word] = vec } }
let d = try JSONSerialization.data(withJSONObject: v)
FileHandle.standardOutput.write(d)
'''


def vectors():
    phrases=list(dict.fromkeys(["missing",*SERVICES,*[v for options in MEANINGS.values() for v in options.values()]]))
    words=sorted(set(" ".join(phrases).split()))
    binary=subprocess.run(["/usr/bin/swift","-module-cache-path","/tmp/codex_semantic_swift_n23",
                           "-e",SWIFT,*words],check=True,capture_output=True)
    bank=json.loads(binary.stdout)
    lookup={}
    for phrase in phrases:
        vec=np.mean([bank[w] for w in phrase.split() if w in bank],axis=0)
        if vec.shape!=(300,) or not np.isfinite(vec).all():
            raise ValueError(f"Incomplete semantic representation: {phrase}")
        lookup[phrase]=vec/np.linalg.norm(vec)
    fitted=PCA(n_components=16,svd_solver="full").fit_transform(np.array(list(lookup.values())))
    return dict(zip(phrases,fitted)),dict(word_dimension=300,observed_words=sorted(bank),
                                          missing_words=sorted(set(words)-set(bank)),os=platform.mac_ver()[0])


def features(x,bank,permuted):
    rng=np.random.default_rng(729)
    d=np.zeros((len(x),16))
    for col,m in MEANINGS.items():
        phrases=list(m.values())
        if permuted:
            # Rotate each alphabet to preserve marginal vocabulary while breaking its semantics.
            phrases=np.roll(phrases,int(rng.integers(1,len(phrases)))).tolist()
        mapping=dict(zip(m,phrases))
        which=x[col].map(mapping).fillna("missing")
        d+=np.stack([bank[word] for word in which])
    block=np.stack([bank[s] for s in SERVICES])
    if permuted:
        block=np.roll(block,2,axis=0)
    d+=np.log1p(x[BILLS].fillna(0).to_numpy())@block/6
    return x.join(pd.DataFrame(d,columns=[f"sem{i}" for i in range(d.shape[1])]))


def run():
    bank,metadata=vectors()
    panel=pd.read_csv(ART/"nov23_global_error_inventory.csv")
    row=panel.row.to_numpy()
    y,b,h=read_panel()
    np.testing.assert_array_equal(panel.target,y)
    wrong=~b.any(0)
    hard=wrong&~h.any(axis=(0,1))
    assert wrong.sum()==1070 and hard.sum()==466
    frame=pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"]).iloc[row]
    x=compact_features(frame).reset_index(drop=True)
    group=frame.PassengerId.str.split("_").str[0].to_numpy()
    variants={"nominal":x,"semantic":features(x,bank,False),"scrambled":features(x,bank,True)}
    out=ART/"nov23_apple_semantics"
    out.mkdir(exist_ok=True)
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            base,fold=z["base"],z["fold"]
            verify(z,row,y,fold)
        pred={}
        scores={}
        for n,f in variants.items():
            p=np.full(len(y),np.nan)
            for k in np.unique(fold):
                q=fold==k
                assert not np.intersect1d(group[q],group[~q]).size
                model=CatBoostClassifier(iterations=500,depth=5,learning_rate=.06,thread_count=2,
                      verbose=False,allow_writing_files=False,random_seed=398,cat_features=CATEGORICAL)
                model.fit(f.iloc[~q],y[~q]);p[q]=model.predict_proba(f.iloc[q])[:,1]
            assert np.isfinite(p).all()
            pred[n]=p
            for key,s in [(n,p),(n+"_blend",.75*base+.25*p)]:
                ok=(s>=.5)==y;bad=(base>=.5)!=y
                scores[key]=dict(accuracy=float(ok.mean()),fixed=int((bad&ok).sum()),harmed=int((~bad&~ok).sum()),
                                 hard1070=int(ok[wrong].sum()),hard466=int(ok[hard].sum()))
        results[str(seed)]=scores
        np.savez_compressed(out/f"{seed}.npz",row=row,y=y,fold=fold,base=base,p=pred["semantic"],**pred)
        print(seed,[(k,round(v["accuracy"],4),v["hard466"]) for k,v in scores.items()],flush=True)
    write(ROOT/"reports/nov23_apple_semantics/result.json",dict(embedding=metadata,results=results))


if __name__=="__main__":
    with threadpool_limits(2):run()
