"""Nyström logistic model with a separate similarity measure for each manifest view."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import relational_error_corrector as ref

ROOT=Path(__file__).resolve().parent
CATEGORIES=["HomePlanet","CryoSleep","Destination","VIP"]
BILLS=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
SEED=20261119


def coordinates(train,frame):
    def ranks(col):
        ref_values=np.sort(train[col].dropna().to_numpy(float))
        values=frame[col].to_numpy(float)
        l=np.searchsorted(ref_values,values,side="left"); r=np.searchsorted(ref_values,values,side="right")
        return np.column_stack([np.where(np.isnan(values),.5,(l+r)/(2*len(ref_values))),np.isnan(values)])
    def indicators(col):
        return np.column_stack([frame[col].eq(v).to_numpy(float) for v in sorted(train[col].unique())])
    passenger=np.column_stack([*(indicators(col) for col in CATEGORIES),ranks("Age"),ranks("GroupSize")])
    location=np.column_stack([indicators("Deck"),indicators("Side"),ranks("CabinNumber"),ranks("CabinOccupants")])
    spending=np.column_stack([*(ranks(col) for col in BILLS),frame[BILLS].gt(0).to_numpy(float)])
    return [passenger,location,spending]


def kernels(coords,landmarks,scales,recipe):
    distances=[cdist(x,c,metric="sqeuclidean")/scale for x,c,scale in zip(coords,landmarks,scales)]
    if recipe=="isotropic":
        return np.exp(-sum(distances)/3)
    components=[.5*(np.exp(-d)+np.exp(-4*d)) for d in distances]
    a,b,c=components
    return (a+b+c+a*b+a*c+b*c+a*b*c)/7


class ViewKernel:
    def __init__(self,recipe):
        self.recipe=recipe

    def fit(self,train,y):
        self.reference=train.copy()
        coords=coordinates(train,train)
        index=np.random.default_rng(SEED).choice(len(y),768,replace=False)
        self.centers=[part[index] for part in coords]
        self.scales=[]
        for centers in self.centers:
            d=cdist(centers,centers,metric="sqeuclidean")
            self.scales.append(float(np.median(d[d>0])))
        matrix=kernels(self.centers,self.centers,self.scales,self.recipe)
        eigenvectors,U=np.linalg.eigh(matrix)
        valid=eigenvectors>1e-7
        self.basis=U[:,valid]/np.sqrt(eigenvectors[valid])
        design=kernels(coords,self.centers,self.scales,self.recipe)@self.basis
        self.model=LogisticRegression(C=10,max_iter=350).fit(design,y)
        return self

    def predict(self,query):
        design=kernels(coordinates(self.reference,query),self.centers,self.scales,self.recipe)@self.basis
        return self.model.predict_proba(design)[:,1]


def counts_frame(frame):
    roster=pd.concat([pd.read_csv(ROOT/"data/raw"/name,usecols=["PassengerId","Cabin"])
                      for name in ["train.csv","test.csv"]],ignore_index=True)
    groups=roster.PassengerId.str.split("_").str[0]
    return frame.assign(GroupSize=groups.map(groups.value_counts()),
                        CabinOccupants=roster.Cabin.map(roster.Cabin.value_counts()))


def run():
    inputs,splits,_,bases,_=ref.overlap.load_authenticated()
    dev,y,raw=inputs["dev"],inputs["y"],bases["rawMitra"]
    frame=counts_frame(inputs["features"]).iloc[dev]
    models={}
    for recipe in ["isotropic","interaction"]:
        p=np.empty(len(dev))
        for a,b in splits:
            p[b]=ViewKernel(recipe).fit(frame.iloc[a],y[a]).predict(frame.iloc[b])
        models[recipe]=p
        models[recipe+"_half_raw"]=(raw+p)/2
    cat=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    np.testing.assert_array_equal(cat.row_id,dev)
    hard=cat.all_recipes_still_wrong.to_numpy(bool)
    results={key:dict(metrics=ref.score(p,y),vs_raw=ref.pair(p,raw,y),hard546=ref.pair(p,raw,y,hard),
                      fold_net=[ref.pair(p[b],raw[b],y[b])["net_correct"] for _,b in splits]) for key,p in models.items()}
    results["interaction_vs_isotropic"]=ref.pair(models["interaction"],models["isotropic"],y)
    path=ROOT/"reports/mixed_view_kernel/result.json"
    ref.write(path,results)
    np.savez_compressed(ROOT/"artifacts/mixed_view_kernel.npz",dev=dev,**models)
    print(json.dumps(results))


if __name__=="__main__":
    with threadpool_limits(3):
        run()
