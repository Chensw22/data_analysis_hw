"""Learn class-conditional latent trees on the whole manifest; marginalize missing fields."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import SEEDS, hard_rows
from error_overlap_diagnostic import load_authenticated
from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
ART = ROOT/"artifacts"
BLOCKS = [["FoodCourt","ShoppingMall"],["VRDeck","Spa","RoomService"],
          ["HomePlanet","Destination","VIP","party_size"],["Deck","Side","CabinNumber"],["Age","CryoSleep"]]


def encode(frame, party_size):
    data = frame.assign(party_size=party_size)
    matrix, widths = {}, {}
    bins = {name:[0,1,100,400,700,1000,1500,2500,5000] for name in BLOCKS[0]+BLOCKS[1]}
    bins.update({"Age":[1,6,12,18,25,35,50,65],"CabinNumber":list(range(150,1950,150)),
                 "party_size":[1,2,3,4,6]})
    for block in BLOCKS:
        for name in block:
            column = data[name]
            if name in bins:
                code = np.searchsorted(bins[name],column.to_numpy(float),side="left")
                code[column.isna()] = -1
                widths[name] = len(bins[name])+1
            else:
                code, levels = pd.factorize(column.replace("__MISSING__",np.nan), sort=True)
                widths[name] = len(levels)
            matrix[name]=code
    return matrix,widths


def normalize(a):
    return a / a.sum(axis=-1,keepdims=True)


class ManifestTree:
    def __init__(self, blocks, widths, states, seed):
        rng = np.random.default_rng(seed)
        self.prior = normalize(np.ones(states))
        self.blocks = blocks
        self.links = [normalize(rng.gamma(1.5,1,(states,states))) for _ in blocks]
        self.emission = {name:normalize(rng.gamma(2.,1,(states,width))) for name,width in widths.items()}

    def infer(self,x,rows):
        group_values, messages, scales = [],[],[]
        for members,link in zip(self.blocks,self.links):
            score = np.zeros((len(rows),len(self.prior)))
            for name in members:
                code=x[name][rows]
                seen=code>=0
                score[seen] += np.log(self.emission[name][:,code[seen]].T)
            peak=score.max(axis=1)
            value=np.exp(score-peak[:,None])
            group_values.append(value)
            messages.append(value @ link.T)
            scales.append(peak)
        root=np.log(self.prior)+sum(np.log(msg) for msg in messages)
        normalizer=logsumexp(root,axis=1)
        return normalizer+sum(scales),np.exp(root-normalizer[:,None]),group_values,messages

    def learn(self,x,rows):
        for _ in range(1 if len(self.prior)==1 else 35):
            _,post,values,msgs=self.infer(x,rows)
            self.prior=normalize(3+post.sum(axis=0))
            for i,members in enumerate(self.blocks):
                link=self.links[i]
                partial=post/msgs[i]
                child=(partial@link)*values[i]
                self.links[i]=normalize(2+link*(partial.T@values[i]))
                for name in members:
                    code=x[name][rows]
                    seen=code>=0
                    counts=np.full_like(self.emission[name],1.5)
                    np.add.at(counts.T,code[seen],child[seen])
                    self.emission[name]=normalize(counts)


def run():
    inp,_,_,_,_=load_authenticated()
    dev,y=inp["dev"],inp["y"].astype(bool)
    roster=pd.concat([pd.read_csv(ROOT/f"data/raw/{s}.csv",usecols=["PassengerId"]).PassengerId
                      for s in ["train","test"]],ignore_index=True)
    party=roster.str.split("_").str[0]
    x,widths=encode(inp["features"].iloc[dev].reset_index(drop=True),party.map(party.value_counts()).to_numpy()[dev])
    hard=hard_rows(dev,y)
    shuffled=np.random.default_rng(2301).permutation(list(widths)).tolist()
    placebo=[]
    for block in BLOCKS:
        placebo.append(shuffled[:len(block)])
        shuffled=shuffled[len(block):]
    out=ART/"nov23_manifest_tree"
    out.mkdir(exist_ok=True)
    result={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as z:
            np.testing.assert_array_equal(z["row"],dev)
            np.testing.assert_array_equal(z["y"],y)
            base,fold=z["base"],z["fold"]
        pred={}
        for name,blocks,states in [("independent",BLOCKS,1),("manifest_tree",BLOCKS,3),("permuted_tree",placebo,3)]:
            p=np.full(len(y),np.nan)
            for k in np.unique(fold):
                t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
                assert not set(party.iloc[dev[t]])&set(party.iloc[dev[q]])
                scores=[]
                for cls in [False,True]:
                    train=t[y[t]==cls]
                    model=ManifestTree(blocks,widths,states,2301+int(cls))
                    model.learn(x,train)
                    density=model.infer(x,q)[0]
                    scores.append(density+np.log(len(train)/len(t)))
                p[q]=expit(scores[1]-scores[0])
            assert np.isfinite(p).all()
            pred[name]=p
        summary={}
        for name,p in pred.items():
            for label,score in [(name,p),(name+"_blend",.75*base+.25*p)]:
                fixed=(base>=.5)!=y
                correct=(score>=.5)==y
                summary[label]=dict(accuracy=float(correct.mean()),fixed=int((fixed&correct).sum()),
                                    harmed=int((~fixed&~correct).sum()),hard447_correct=int(correct[hard].sum()))
        np.savez_compressed(out/f"{seed}.npz",row=dev,y=y,fold=fold,base=base,**pred)
        result[str(seed)]=summary
        print(seed,summary,flush=True)
    write(ROOT/"reports/nov23_manifest_tree/result.json",dict(results=result,blocks=BLOCKS,placebo=placebo))


if __name__=="__main__":
    with threadpool_limits(2):
        run()
