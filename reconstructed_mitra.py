"""Fit the counts Mitra model on a consistently reconstructed train/test manifest."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"

import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import torch

import mitra_counts_fixed15_online_first as counts
from exaone_validation_data import compact_features
from kaggle_workflow import sha256, validate_submission
from peer_cabin_impute import infer

ROOT = Path(__file__).resolve().parent
NAME = "proposal_v111_reconstructed_mitra"
VERSION = "v111_Mitra_counts_reconstructed_manifest"
BILLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def reconstruct():
    frames = [pd.read_csv(ROOT/"data/raw"/f,usecols=lambda c:c not in ("Transported","Name"))
              for f in ("train.csv","test.csv")]
    original = pd.concat(frames,ignore_index=True)
    cleaned = original.copy()
    groups = original.PassengerId.str.split("_").str[0]
    planet = original.HomePlanet.groupby(groups)
    cleaned.HomePlanet = original.HomePlanet.fillna(groups.map(planet.first().where(planet.nunique().eq(1))))
    paid = original[BILLS].gt(0).any(axis=1)
    cleaned.loc[original.CryoSleep.isna() & paid,"CryoSleep"] = False
    sleep_or_child = original.CryoSleep.eq(True) | original.Age.lt(13)
    cleaned.loc[sleep_or_child,BILLS] = original.loc[sleep_or_child,BILLS].fillna(0)
    _, patches, _ = infer(test_only=False)
    cleaned.loc[patches.index,"Cabin"] = (patches.Deck + "/" + patches.CabinNumber.astype(int).astype(str) + "/" + patches.Side)
    keys = ["HomePlanet","CryoSleep","Cabin",*BILLS]
    filled = original[keys].isna() & cleaned[keys].notna()
    # Counts retain observed cabin occupancy. An interpolated cabin number is not evidence of shared occupancy.
    roster = counts.RosterCounts(original,compact_features(cleaned))
    stats = {k:{column:int(value) for column,value in rows.sum().items()} for k,rows in
             [("training",filled.iloc[:8693]),("test",filled.iloc[8693:])]}
    return roster,stats


def fit(roster, y, output):
    device = "mps"
    counts.base.seed_everything(counts.SEED,device)
    x = roster.plain.iloc[:len(y)]
    ids = np.arange(len(y), dtype=np.int64)
    prep = counts.CountsPreprocessor(roster).fit(x,y,counts.SEED)
    support = prep.transform(x)
    assert support.shape == (8693,15) and prep.active_columns == counts.COLUMNS
    model,config = counts.training.fresh_model(device)
    data = torch.as_tensor(support,device=device)
    labels = torch.as_tensor(y,device=device)
    optimizer = counts.base.optimizer_for(model)
    rng = np.random.default_rng(counts.SEED)
    history = []
    for step in range(1,counts.STEPS+1):
        counts.require_false()
        history.append(counts.training.episode_step(model,optimizer,data,labels,ids,rng,step))
        counts.write(output/"training_log.json",dict(step=step,history=history))
        print(f"Mitra reconstructed update {step}/{counts.STEPS}",flush=True)
    del optimizer,data,labels
    counts.base.release_memory()
    meta = dict(candidate=VERSION,training_rows=len(y),seed=counts.SEED,refit_optimizer_steps=len(history))
    return counts.CountsPredictor(prep,config,model.state_dict(),support,y,device,meta,query_chunk=512)


def export():
    counts.verify_reference()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal device unavailable")
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(.65)
    out = ROOT/"artifacts"/NAME
    report = ROOT/"reports"/NAME
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    counts.base.DEADLINE = started+7200
    roster,stats = reconstruct()
    train = pd.read_csv(ROOT/"data/raw/train.csv", usecols=["Transported","PassengerId"])
    np.testing.assert_array_equal(train.PassengerId,roster.passenger_ids.iloc[:len(train)])
    test_ids = roster.passenger_ids.iloc[len(train):]
    np.testing.assert_array_equal(test_ids,pd.read_csv(ROOT/"data/raw/test.csv",usecols=["PassengerId"]).PassengerId)
    model = fit(roster, train.Transported.to_numpy(dtype=np.int64), out)
    p = model.predict_proba_by_passenger_ids(test_ids)[:,1]
    joblib.dump(model,out/"model.joblib")
    model.release()
    del model
    counts.base.release_memory()
    restored = joblib.load(out/"model.joblib")
    restored.device = "mps"
    q = restored.predict_proba_by_passenger_ids(test_ids)[:,1]
    restored.release()
    np.testing.assert_array_equal(p,q)
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    pd.DataFrame(dict(PassengerId=test_ids,probability=p,reload_probability=q)).to_csv(out/"test_probabilities.csv",index=False)
    pd.DataFrame(dict(PassengerId=test_ids,Transported=p>=.5)).to_csv(out/"submission.csv",index=False)
    check = validate_submission(out/"submission.csv",ROOT/"data/raw/test.csv")
    old = pd.read_csv(ROOT/"artifacts/proposal_v108_party_bills/submission.csv")
    np.testing.assert_array_equal(old.PassengerId,test_ids)
    meta = dict(**check,version=VERSION,model_sha256=sha256(out/"model.joblib"),reload_difference=0,
                changed_vs_v108=int(((p>=.5)!=old.Transported).sum()),imputed_cells=stats,
                seconds=time.monotonic()-started,recipe_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    for folder in [out,report]:
        counts.write(folder/"manifest.json",meta)
        counts.write(folder/"selection.json",dict(version=VERSION,threshold=.5))
    print(meta,flush=True)


if __name__ == "__main__":
    export()
