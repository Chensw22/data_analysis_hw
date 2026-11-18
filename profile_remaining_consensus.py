"""Describe which historic hard cases survived five new independent predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def run():
    catalog=pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv",dtype={"PassengerId":str})
    roster=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId","HomePlanet","CryoSleep",
                                                         "RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])
    rows=roster.iloc[catalog.row_id]
    np.testing.assert_array_equal(rows.PassengerId,catalog.PassengerId)
    y=catalog.y.to_numpy()
    hard=catalog.all_recipes_still_wrong.to_numpy(bool)
    remaining=hard.copy()
    for file,ids,key in [("hard_weighted_combo_oof.npz","dev_ids","weighted_cat"),
            ("mechanism_combo_empty_query_fix/oof.npz","dev_ids","mechanism"),
            ("top3_stage/density_oof.npz","dev","density"),("top3_stage/split_views.npz","dev","views")]:
        with np.load(ROOT/"artifacts"/file) as packet:
            np.testing.assert_array_equal(packet[ids],catalog.row_id)
            if key=="weighted_cat":
                raw=packet["raw"].copy()
            decisions=packet[key].reshape(len(y),-1)>=.5
            remaining &= np.all(decisions!=y[:,None],axis=1)
    assert remaining.sum()==425
    asleep=rows.CryoSleep.eq(True).to_numpy()
    earth=rows.HomePlanet.eq("Earth").to_numpy()
    result={}
    for name,mask in [("all_historical",hard),("correct_by_at_least_one",hard&~remaining),
                      ("remaining",remaining)]:
        result[name]=dict(rows=int(mask.sum()),actual_false=int((mask&(y==0)).sum()),
                          actual_true=int((mask&(y==1)).sum()),earth_sleep_false=int((mask&earth&asleep&(y==0)).sum()),
                          completely_zero_bills=int((mask&rows.iloc[:,rows.columns.isin(["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"])].eq(0).all(axis=1).to_numpy()).sum()),
                          raw_wrong_confidence_median=float(np.median(np.maximum(raw,1-raw)[mask])))
    path=ROOT/"reports/top3_stage/remaining_profile.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    run()
