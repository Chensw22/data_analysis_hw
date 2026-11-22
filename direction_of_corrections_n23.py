"""Evaluate whole-population corrections by observable spending regime and direction of change."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import ART, ROOT, SEEDS, read_panel
from relational_error_corrector import write


def run():
    row=pd.read_csv(ART/"nov23_global_error_inventory.csv").row.to_numpy()
    y, old, earlier=read_panel()
    hard=~old.any(0)&~earlier.any(axis=(0,1))
    bills=["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    raw=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId",*bills]).iloc[row]
    group=raw.PassengerId.str.split("_").str[0].to_numpy()
    spending=np.select([raw[bills].gt(0).any(axis=1),raw[bills].isna().any(axis=1)],
                       ["positive","incomplete"],default="zero")
    results={}
    for seed in SEEDS:
        with np.load(ART/f"nov23_teacher_field_offsets/{seed}.npz") as z:
            base, fold, proposal=z["base"]>=.5,z["fold"],z["offset3"]>=.5
            verify(z,row,y,fold)
        cells=np.char.add(np.char.add(spending,"/"),np.where(proposal,"predict_transported","predict_not_transported"))
        changed=proposal!=base
        correct=proposal==y
        accept=np.zeros(len(y),bool)
        for k in np.unique(fold):
            pool=np.flatnonzero(fold==k)
            for a,b in GroupKFold(3,shuffle=True,random_state=117).split(pool,groups=group[pool]):
                t,q=pool[a],pool[b]
                assert not np.intersect1d(group[t],group[q]).size
                for cell in np.unique(cells):
                    observed=t[changed[t]&(cells[t]==cell)]
                    if correct[observed].sum()>len(observed)/2:
                        accept[q]|=changed[q]&(cells[q]==cell)
        counts={}
        for cell in np.unique(cells):
            mask=changed&(cells==cell)
            counts[cell]=dict(proposals=int(mask.sum()),fixed=int((mask&correct).sum()),
                              harmed=int((mask&~correct).sum()),fixed_of466=int((mask&hard&correct).sum()))
        final=np.where(accept,proposal,base)
        results[seed]=dict(whole_dev_diagnostic=counts, booking_held_out_policy=dict(
            accuracy=float((final==y).mean()),fixed=int((accept&correct).sum()),harmed=int((accept&~correct).sum()),
            fixed_of466=int((accept&correct&hard).sum())))
    write(ROOT/"reports/nov23_correction_direction/result.json",dict(results=results))
    print(results)


if __name__=="__main__":
    run()
