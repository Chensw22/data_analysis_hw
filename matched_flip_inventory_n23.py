"""Measure error targeting at each model's actual number and difficulty of proposed changes."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART,ROOT,SEEDS,read_panel
from relational_error_corrector import write

FAMILIES={"teacher_field_offsets":"offset3","gibbs_manifest":"adaptive","manifest_tree":"manifest_tree",
          "sparse_rules":"rules","tabicl_context":"global_context","party_hotdeck":"multiple",
          "expression_search":"selected","class_analogues":"class_linear","spend_copula":"conditional_copula",
          "pair_pretext":"integrity","passenger_sampler":"sequential","exact_booking_field":"joint",
          "conditional_calibration":"context","residual_agreement":"agreement_referee"}


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y,hits,others=read_panel()
    hard=~hits.any(0)&~others.any(axis=(0,1))
    persistent=~hits.any(0)
    bills=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId",*bills])
    parties=raw.PassengerId.str.split("_").str[0]
    size=parties.map(parties.value_counts()).iloc[row].to_numpy()
    raw=raw.iloc[row]
    spending=np.select([raw[bills].gt(0).any(axis=1),raw[bills].isna().any(axis=1)],[2,1],default=0)
    results={}
    differences=[]
    for seed in SEEDS:
        with np.load(ART/f"nov23_group_kernel/{seed}.npz") as ref:
            fold,base=ref["fold"],ref["base"]
            verify(ref,row,y,fold)
        wrong=(base>=.5)!=y
        # Score intervals 0.00-0.05, ..., 0.95-1 include both direction and confidence.
        risk=np.floor(20*base).astype(int).clip(0,19)
        strata=np.column_stack([fold,risk,spending,np.minimum(size,3)])
        _,cell=np.unique(strata,axis=0,return_inverse=True)
        n=np.bincount(cell)
        info={}
        advantage=[]
        for folder,key in FAMILIES.items():
            with np.load(ART/f"nov23_{folder}/{seed}.npz") as z:
                verify(z,row,y,fold)
                np.testing.assert_allclose(base,z["base"])
                flip=(z[key]>=.5)!=(base>=.5)
            # Randomly reassign the exact count of flips inside each cell.
            # This is the exact expected passenger-level outcome, with no simulation.
            propensity=np.bincount(cell,weights=flip,minlength=len(n))[cell]/n[cell]
            np.testing.assert_allclose(propensity.sum(),flip.sum())
            actual=int((flip&wrong).sum())
            expected=float(propensity@wrong)
            advantage.append((flip.astype(float)-propensity)*wrong)
            info[folder]=dict(proposed=int(flip.sum()),fixed=actual,harmed=int(flip.sum()-actual),
                              expected_fixed_matched=expected,added_fixes=float(actual-expected),
                              persistent1070=int(flip[persistent].sum()),matched1070=float(propensity[persistent].sum()),
                              unresolved466=int(flip[hard].sum()),matched466=float(propensity[hard].sum()))
        differences.append(np.array(advantage))
        results[seed]=info
    # Mean number of extra correct proposals across splits, resampling bookings
    # together. Repeating a passenger across splits does not add observations.
    _,g=np.unique(parties.iloc[row],return_inverse=True)
    delta=np.mean(differences,axis=0).T
    b=np.zeros((g.max()+1,len(FAMILIES)))
    np.add.at(b,g,delta)
    rng=np.random.default_rng(6391)
    draws=[]
    for _ in range(40):
        count=rng.multinomial(len(b),np.full(len(b),1/len(b)),size=50)
        draws.append(count@b)
    ci=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
    mean={name:dict(mean_added_fixes_per_6954=float(delta[:,i].sum()),paired_95_added_fixes=ci[:,i].tolist())
          for i,name in enumerate(FAMILIES)}
    write(ROOT/"reports/nov23_matched_flips/result.json",dict(results=results,mean=mean,
          definition="Expected number repaired by random reallocation of each model's flips among passengers with the same fold, baseline probability interval (width .05), spending state, and booking size 1/2/3+. Counts conditional on reused observations and model scores, not a deployable selector. An extra fix also avoids a harm at fixed flip count."))
    for name in FAMILIES:
        print(name,[round(results[s][name]["added_fixes"],1) for s in SEEDS],
              [(results[s][name]["unresolved466"],round(results[s][name]["matched466"],1)) for s in SEEDS])


if __name__=="__main__":
    with threadpool_limits(2):
        run()
