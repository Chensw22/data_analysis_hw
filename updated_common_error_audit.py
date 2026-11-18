"""Count common-error coverage and collateral damage across independent prediction mechanisms."""
from pathlib import Path
import numpy as np
import pandas as pd

from relational_error_corrector import write

ROOT = Path(__file__).resolve().parent
# One explicitly identified output per mechanism, including the five earlier single models.
PACKETS = [
    ("weighted tree", "hard_weighted_combo_oof.npz","weighted_cat"),
    ("spending mechanism", "mechanism_combo_empty_query_fix/oof.npz","mechanism"),
    ("conditional dependence tree","top3_stage/density_oof.npz","density"),
    ("passenger and billing views","top3_stage/split_views.npz","views"),
    ("manifest surprise","manifest_surprise/oof.npz","augmented"),
    ("class-conditional bills","bill_likelihood_oof.npz","bill_posterior"),
    ("mixed kernels","mixed_view_kernel.npz","interaction"),
    ("LimiX 512","limix2_passenger_trial/oof.npz","probability"),
    ("Deep Sets","deepsets_peer_oof.npz","peers"),
    ("pseudo-label student","query_pseudolabel_oof.npz","pseudo_student"),
    ("unknown sleep marginal","cryo_marginal_oof.npz","marginal"),
    ("counts Mitra","v96_oof_completion/oof.npz","prob"),
]


def audit():
    catalog = pd.read_csv(ROOT/"artifacts/error_rescue_20261117/catalog_rows.csv")
    dev = catalog.row_id.to_numpy()
    hard = catalog.all_recipes_still_wrong.to_numpy(bool)
    y = catalog.y.to_numpy()
    x = pd.read_csv(ROOT/"data/raw/train.csv",usecols=lambda c:c not in ["Name","Transported"]).iloc[dev]
    np.testing.assert_array_equal(x.PassengerId,catalog.PassengerId)
    names,columns = [],[]
    for name,file,key in PACKETS:
        with np.load(ROOT/"artifacts"/file,allow_pickle=False) as packet:
            np.testing.assert_array_equal(packet["dev" if "dev" in packet else "dev_ids"],dev)
            if not columns:
                raw = packet["raw"]>=.5
            values = packet[key].reshape(len(y),-1)
            assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
            for j, p in enumerate(values.T):
                names.append(name if values.shape[1]==1 else f"{name} {j+1}")
                columns.append(p>=.5)
    decisions = np.column_stack(columns)
    correct = decisions==y[:,None]
    hits = correct.sum(axis=1)
    assert hard.sum()==546 and np.sum(hard & ~correct[:,:5].any(axis=1))==425
    buckets={}
    for name,mask in [("none",hits==0),("one",hits==1),("at_least_two",hits>=2),("majority",hits>len(columns)/2)]:
        m=hard&mask
        buckets[name]=dict(rows=int(m.sum()),actual_false=int(((y==0)&m).sum()),
            earth_sleep_false=int((x.HomePlanet.eq("Earth")&x.CryoSleep.eq(True)&(y==0)&m).sum()),
            complete_attributes=int((x.drop(columns="PassengerId").notna().all(axis=1)&m).sum()))
    thresholds={}
    dissent = (decisions!=raw[:,None]).sum(axis=1)
    for votes in [1,2,len(columns)//2+1,len(columns)]:
        switch=dissent>=votes
        thresholds[str(votes)]=dict(dissent_rows=int(switch.sum()),raw_errors_corrected=int((switch&(raw!=y)).sum()),
             raw_correct_lost=int((switch&(raw==y)).sum()),historical_hard_corrected=int((switch&hard).sum()))
    new=~correct[:,:5].any(axis=1)&correct[:,5:].any(axis=1)
    result=dict(models=len(names),hard546=buckets,additional_vs_first_five=int((new&hard).sum()),
        unique_among_all={name:int((hard&correct[:,i]&(hits==1)).sum()) for i,name in enumerate(names)},
        observed_disagreement_rules=thresholds,limitations="Same adaptive development folds. Fixed list is not every historical experiment. Correctness is diagnostic only.")
    write(ROOT/"reports/common_error_followup/coverage.json",result)
    print(result)


if __name__=="__main__":
    audit()
