"""Replay three released correction policies from cached development and submission inputs."""
import json

import joblib
import numpy as np
import pandas as pd

from error_overlap_diagnostic import FAMILIES, load_authenticated
from export_consensus_routes import BASE, HISTORICAL, official_decisions
from export_majority_specialist import decide, probabilities
from export_nov21_error_batch import test_probabilities
from kaggle_workflow import sha256
from released_correction_audit_n23 import ART, ROOT, REPORT as RECEIPTS, development, release
from relational_error_corrector import write


def rules(p, hist, parent, specialist, second):
    anchor = p >= .5
    agree = ((hist >= .5) == anchor[:, None]).sum(1)
    locked = parent != anchor
    maj = np.where(locked, parent, anchor ^ ((agree < 3) | ((agree == 5) & specialist)))
    soft = (hist.sum(1)+p)/6 >= .5
    return {116:maj,117:np.where(agree==5,maj,anchor),
            121:np.where(locked|(agree==5),maj,soft),
            122:np.where((agree==0)&(second!=maj),parent,maj)},agree


def main():
    inputs,splits,fold,models,_=load_authenticated()
    y,pd_local,_=development()
    rows=inputs["dev"]
    np.testing.assert_array_equal(y,inputs["y"])
    def cache(path,key):
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z["dev"] if "dev" in z else z["dev_ids"],rows)
            np.testing.assert_array_equal(z[next(k for k in ("fold","folds","fold_id") if k in z)],fold)
            return z[key]
    p=cache("v108_exact_oof/oof.npz","v108")
    hist=np.column_stack([models[k] for k in FAMILIES])
    parent=pd_local[115]
    special=cache("consensus_dispute_specialist.npz","agreement_trained_five_agree")!=parent
    second=cache("sequential_rescue_local.npz","member").mean(1)[:,2]>=.5
    local,agreement=rules(p,hist,parent,special,second)
    for v in local:np.testing.assert_array_equal(local[v],pd_local[v])
    ids=pd.read_csv(ROOT/"data/raw/test.csv",usecols=["PassengerId"]).PassengerId
    official=json.loads((RECEIPTS/"primary_receipts.json").read_text())["scores"]
    releases={v:release(v,ids,official) for v in (108,115,116,117,121,122)}
    def score(v):
        return int(round(len(ids)*releases[v][0]))
    folders={v:next((ROOT/"reports").glob(f"proposal_v{v}_*")) for v in releases}
    manifest=json.loads((folders[116]/"manifest.json").read_text())
    gates_path=ART/folders[116].name/"gates.joblib"
    assert sha256(gates_path)==manifest["model_sha256"]
    for v in [121,122]:
        sources=json.loads((folders[v]/"manifest.json").read_text())["component_sources"]
        assert all(sha256(ROOT/name)==h for name,h in sources.items())
    qt=probabilities(ART/BASE/"test_probabilities.csv",ids)
    np.testing.assert_array_equal(qt>=.5,releases[108][1])
    members=np.column_stack([test_probabilities(name,ids) for name in HISTORICAL])
    for i,name in enumerate(HISTORICAL):
        np.testing.assert_array_equal(members[:,i]>=.5, official_decisions(name,ids)[0])
    with np.load(ART/"full_challenger_margin/test_components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"],ids)
        np.testing.assert_array_equal(z["v108"],qt)
        np.testing.assert_array_equal(z["members"],z["reload_members"])
        replay,risk,_=decide(qt,releases[115][1],members>=.5,members[:,3],z["members"],joblib.load(gates_path))
        np.testing.assert_array_equal(replay,releases[116][1])
    with np.load(ART/"nov21_specialists/components.npz") as z:
        np.testing.assert_array_equal(z["passenger_ids"],ids)
        second=z["voters"][:,2]>=.5
    test,ta=rules(qt,members,releases[115][1],risk>=.5,second)
    for v in test:np.testing.assert_array_equal(test[v],releases[v][1])
    groups=np.asarray(inputs["groups"])
    cold=np.zeros(len(y),bool)
    for t,q in splits:cold[q]=~np.isin(groups[q],groups[t])
    test_groups=ids.str.split("_").str[0]
    train_groups=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.str.split("_").str[0]
    assert not set(test_groups)&set(train_groups)
    contrasts={}
    for a,b in [(108,115),(115,116),(108,117),(117,116),(116,121),(116,122),
                (108,116),(108,121),(108,122)]:
        good,before=pd_local[b]==y,pd_local[a]==y
        delta=good.astype(int)-before
        changed=pd_local[b]!=pd_local[a]
        tm=releases[b][1]!=releases[a][1]
        rec=dict(local_changes=int(changed.sum()),local_net=int(delta.sum()),test_changes=int(tm.sum()),
                 public_net=score(b)-score(a),cold_changes=int(changed[cold].sum()),cold_net=int(delta[cold].sum()),
                 local_net_per_fold=[int(delta[q].sum()) for _,q in splits],
                 test_changes_by_agreement=np.bincount(ta[tm],minlength=6).tolist(),
                 development_by_agreement=[])
        for k in range(6):
            cell=agreement==k
            rec["development_by_agreement"].append(dict(local_changed=int((changed&cell).sum()),
                                                       local_net=int(delta[cell].sum())))
        # Use only observable properties of changed rows for standardization.
        # This estimates transport under stable stratum-level repair precision.
        development_cell = agreement*4 + 2*(p>=.5) + (parent!=pd_local[108])
        test_cell = ta*4 + 2*(qt>=.5) + (releases[115][1]!=releases[108][1])
        for label,eligible in (("all",np.ones(len(y),bool)),("cold",cold)):
            expected, matched, min_rows = 0.,0,len(y)
            for cell in np.unique(test_cell[tm]):
                target_count=int((tm & (test_cell==cell)).sum())
                source=eligible & changed & (development_cell==cell)
                if source.any():
                    expected+=target_count*delta[source].mean()
                    matched+=target_count
                    min_rows=min(min_rows,int(source.sum()))
            rec["placement_"+label]=dict(expected_net=float(expected),matched_test=matched,
                                         min_source_rows=min_rows)
        contrasts[f"{a}-{b}"]=rec
    member_scores={}
    for j,(k,name) in enumerate(zip(FAMILIES,HISTORICAL)):
        file=ART/name/"submission.csv"
        matches=[json.loads(p.read_text()) for p in (ROOT/"reports"/name/"online_attempts").glob("*.json")]
        receipt=next(r for r in matches if r["file_sha256"]==sha256(file) and r["evaluation"]["Status"]=="COMPLETE")
        member_scores[k]=dict(local=float(((hist[:,j]>=.5)==y).mean()),public=float(receipt["evaluation"]["Public Score"]))
    out=dict(local_rows=len(y),test_rows=len(ids),unseen_local=int(cold.sum()),unseen_test=len(ids),
             verified_versions=list(local),members=member_scores,local_v108=float((pd_local[108]==y).mean()),
             public_v108=releases[108][0],contrasts=contrasts)
    write(ROOT/"reports/nov23_release_parity/results.json",out)
    for k,r in contrasts.items():print(k,r["local_net"],r["public_net"],r["cold_changes"],r["cold_net"],r["test_changes"],r["test_changes_by_agreement"])
    print(member_scores,out["local_v108"],cold.sum())
if __name__=="__main__":
    main()
