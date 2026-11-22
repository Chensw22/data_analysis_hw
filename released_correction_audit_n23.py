"""Audit public release scores against OOF correction coverage on the same 6,954 passengers."""
from hashlib import sha256
from itertools import combinations
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from broad_partition_evidence_n23 import ART,ROOT,historical,write
from exact_fold_error_selector import cached
from error_trained_experts import EXPERTS
from relational_error_corrector import overlap
from residual_vote_nov21 import decide
from sleeper_manifest_alternatives import decisions,relative_risk
from route_consensus_errors import FAMILIES

REPORT=ROOT/"reports/nov23_released_correction_audit"


def development():
    inp,_,fold,models,_=overlap.load_authenticated()
    dev,y=inp["dev"],inp["y"].astype(bool)
    get=lambda folder,key:cached(folder+".npz",key,dev,fold)
    base=get("v108_exact_oof/oof","v108")>=.5
    route=lambda key:get("consensus_error_routes",key)
    composed=lambda key:get("composed_local_error_routes",key)
    pred={108:base,114:base^route("three_challengers"),115:base^route("combined_by_historical_agreement"),
          116:composed("five_family_majority_with_specialist")}
    agree=(np.column_stack([models[k]>=.5 for k in FAMILIES])==base[:,None]).all(1)
    pred[117]=np.where(agree,pred[116],base)
    for number,key in [(118,"either"),(119,"committee"),(120,"tree")]:
        pred[number]=get("replicate_error_committee","original_"+key)
    pred[121]=composed("five_family_plus_v108_soft_vote_with_specialist")
    pred[122]=get("confirm_minority_exceptions","second_confirms")
    pred[123]=get("conditional_donor_errors","matched_donors")
    for n,k in ((124,"first_only"),(125,"sequential"),(126,"parallel")):
        pred[n]=get("sequential_rescue_local",k)
    pred[127]=get("sleeper_consensus_scope","original")
    pred[128],alternatives=decisions(base,pred[127],get("sleeper_manifest_risk","manifest_party_a"),
            get("sleeper_manifest_risk","manifest_berth_party_a"),get("sleeper_consensus_scope","eligible"),
            relative_risk(inp["features"].iloc[dev],y,np.asarray(inp["groups"]))["party_a"])
    pred.update({n:alternatives["v"+str(n)] for n in (129,130)})
    ids=pd.read_csv(ROOT/"data/raw/train.csv",usecols=["PassengerId"]).PassengerId.to_numpy()[dev]
    def table(file,label,by_id=True):
        f=pd.read_csv(ART/file)
        np.testing.assert_array_equal(f.PassengerId if by_id else f.row,ids if by_id else dev)
        np.testing.assert_array_equal(f.fold,fold)
        np.testing.assert_array_equal(f[label],y)
        return f
    party=table("nov21_party_negative/oof.csv","target",False)
    np.testing.assert_array_equal(party.v128,pred[128])
    pred[131],pred[132]=party.family_consensus.to_numpy(),party.family_contrast.to_numpy()
    matched=get("error_trained_experts","experts")[:,EXPERTS.index("matched_party")]>=.5
    votes=decide(base,pred[128],route("margin_selector"),matched!=base,route("three_challengers"),
                 pred[119]!=base,get("conditional_donor_errors","matched_donors_risk").mean(1)>=.5)
    pred.update({n:votes["v"+str(n)] for n in (133,134)})
    berth=table("nov21_case_checks/berth_local_rows.csv","target")
    final=table("nov21_case_checks/v136_final_oof.csv","true")
    np.testing.assert_array_equal(final.v131,pred[131])
    np.testing.assert_array_equal(final.v128,pred[128])
    np.testing.assert_array_equal(berth.incumbent,pred[128])
    pred[135],pred[136]=berth.v135d.to_numpy(),final.v136_final.to_numpy()
    rows,labels,masks=historical()
    np.testing.assert_array_equal(rows,dev);np.testing.assert_array_equal(labels,y)
    return y,pred,masks


def release(number,ids,official):
    folder,=list((ROOT/"reports").glob(f"proposal_v{number}_*"))
    receipt,=list((folder/"online_attempts").glob("*.json"))
    r=json.loads(receipt.read_text())
    manifest=json.loads((folder/"manifest.json").read_text())
    file=ART/folder.name/"submission.csv"
    digest=sha256(file.read_bytes()).hexdigest()
    assert digest==r["file_sha256"]==manifest["sha256"] and r["version"]==manifest["version"]
    ref=r["submission_ref"]
    assert r["evaluation"]["Status"]=="COMPLETE" and r["evaluation"]["Submission Ref"]==str(ref)
    score=float(r["evaluation"]["Public Score"])
    assert score==float(official[str(ref)])
    frame=pd.read_csv(file)
    np.testing.assert_array_equal(frame.PassengerId,ids)
    assert len(ids)==r["rows"]==manifest["rows"]==4277
    return score,frame.Transported.to_numpy(bool),str(receipt.relative_to(ROOT)),str(ref)


def compare(rows):
    out={}
    public=np.array([r["public"] for r in rows])
    for metric in ("local","repeat_correct","unresolved_correct","test_changes"):
        x=np.array([r[metric] for r in rows])
        pairs=[int(np.sign(x[i]-x[j])*np.sign(public[i]-public[j])) for i,j in combinations(range(len(x)),2)]
        out[metric]=dict(spearman=float(spearmanr(x,public).statistic),agreed=pairs.count(1),
                         reversed=pairs.count(-1),ties=pairs.count(0))
    return out


def main():
    y,pred,masks=development()
    ids=pd.read_csv(ROOT/"data/raw/test.csv",usecols=["PassengerId"]).PassengerId
    official=json.loads((REPORT/"primary_receipts.json").read_text())["scores"]
    releases={n:release(n,ids,official) for n in pred}
    rows=[]
    for n in range(114,137):
        anchor=108 if n<129 else 128
        p=pred[n];q=pred[anchor];right=p==y;good=q==y
        assert p.dtype==bool
        score,test,path,ref=releases[n]
        rows.append(dict(version=n,anchor=anchor,local=float(right.mean()),public=score,receipt=path,reference=ref,
              fixed=int((right&~good).sum()),harmed=int((~right&good).sum()),
              public_gain_pp=round(100*(score-releases[anchor][0]),3),
              test_changes=int((test!=releases[anchor][1]).sum()),
              repeat_correct=int((masks["repeat"]&right).sum()),unresolved_correct=int((masks["unresolved"]&right).sum())))
    # Independent archived report to catch a swapped development recipe.
    md=(ROOT/"reports/nov21_public_evaluations/RESULTS.md").read_text()
    for n,number in re.findall(r'^\| v(\d+) \| [^|]+ \| ([\d.]+)%',md,re.M):
        assert abs(100*(pred[int(n)]==y).mean()-float(number))<.00051
    md=(ROOT/"reports/nov21_quota_close/RESULTS.md").read_text()
    for n,f,h in re.findall(r'^\| v(\d+) \| [^|]+ \| (\d+) / (\d+)',md,re.M):
        rec=next(r for r in rows if r["version"]==int(n))
        assert (rec["fixed"],rec["harmed"])==(int(f),int(h))
    for r in rows:
        anchor=r["anchor"]
        r["repeat_gain"]=r["repeat_correct"]-int((masks["repeat"]&(pred[anchor]==y)).sum())
    historical_right = np.column_stack([pred[n] == y for n in sorted(pred)])
    coverage = {}
    populations = {"all": np.ones(len(y), bool), **masks}
    for name, mask in populations.items():
        count = historical_right[mask].sum(axis=1)
        coverage[name] = dict(passengers=int(mask.sum()), no_release_correct=int((count == 0).sum()),
                              single_release_correct=int((count == 1).sum()), any_release_correct=int((count > 0).sum()))
    write(REPORT/"results.json",dict(rows=rows,coverage=coverage,panels={k:compare(v) for k,v in {
          "108_anchored":rows[:15],"128_anchored":rows[15:],"all":rows}.items()}))
    print([(r["version"],r["fixed"]-r["harmed"],r["repeat_gain"],r["public_gain_pp"]) for r in rows])
    print(compare(rows[:15]),compare(rows[15:]))


if __name__=="__main__":main()
