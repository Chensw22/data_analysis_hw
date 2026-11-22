"""Compare repeatable rescues with a matched random control that preserves each three-split decision pattern."""
import numpy as np
import pandas as pd

from population_rescue_map_n23 import ART,ROOT,SEEDS,read_panel
from grouped_five_way_coverage_n23 import verify
from relational_error_corrector import write


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    persistent=~hits.any(0)
    bills=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId",*bills])
    g=raw.PassengerId.str.split("_").str[0]
    size=g.map(g.value_counts()).iloc[row].to_numpy()
    x=raw.iloc[row][bills]
    spend=np.select([x.gt(0).any(axis=1),x.isna().any(axis=1)],[2,1],default=0)
    flip=[]
    priors=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_teacher_field_offsets/{seed}.npz") as z:
            fold,base=z["fold"],z["base"]
            verify(z,row,y,fold)
            priors.append(base)
            flip.append((z["offset3"]>=.5)!=(base>=.5))
    prior=np.stack(priors)
    nflip=np.sum(flip,axis=0)
    pattern=(prior>=.5).T@np.array([1,2,4])
    median=np.floor(np.median(prior,axis=0)*20).astype(int).clip(0,19)
    cells=np.column_stack([pattern,median,spend,np.minimum(size,3)])
    _,cell=np.unique(cells,axis=0,return_inverse=True)
    results={}
    for name,scope in [("persistent1070",persistent),("unresolved466",hard)]:
        assert np.all(~hits[:,scope])
        rng=np.random.default_rng(173)
        expected=np.zeros(2)
        samples=np.zeros((4000,2),int)
        for k in np.unique(cell[scope]):
            all_rows=cell==k
            chosen=int((scope&all_rows).sum())
            # The exact passenger-level 3-vector is reassigned as a block. For
            # always-wrong baseline rows its joint endpoints are any/all flips.
            n=np.bincount(nflip[all_rows],minlength=4)
            expected+=chosen*np.array([n[1:].sum(),n[3]])/n.sum()
            draw=rng.multivariate_hypergeometric(n,chosen,size=len(samples))
            samples+=np.column_stack([draw[:,1:].sum(1),draw[:,3]])
        observed=[int((nflip[scope]>0).sum()),int((nflip[scope]==3).sum())]
        results[name]=dict(rows=int(scope.sum()),observed_ever_and_all=observed,
                           expected_matched_ever_and_all=expected.tolist(),
                           random_reference_range95=np.quantile(samples,[.025,.975],axis=0).T.tolist())
    write(ROOT/"reports/nov23_matched_flips/repeatability.json",dict(results=results,
          definition="The residual model was not part of the diagnostic error masks. All three flip decisions are reassigned together within observable spending state, size 1/2/3+, three baseline class decisions and 0.05-wide median baseline probability bin. Random ranges characterize this passenger-level control only."))
    print(results)


if __name__=="__main__":
    run()
