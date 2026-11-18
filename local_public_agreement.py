"""Compare exact version-matched historical development and public scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau

from relational_error_corrector import write

ROOT=Path(__file__).resolve().parent


def run():
    tiers=pd.read_csv(ROOT/"reports/method_tiers.csv").set_index("version")
    scores=tiers.accuracy.to_dict()
    v96=json.loads((ROOT/"reports/v96_oof_completion/result.json").read_text())
    scores[v96["version"]]=v96["accuracy"]
    matches=[]
    for f in sorted((ROOT/"reports").glob("*/online_attempts/*.json")):
        row=json.loads(f.read_text())
        if row.get("evaluation",{}).get("Status")!="COMPLETE":continue
        version=row["version"]
        if version not in scores:continue
        matches.append(dict(version=version,development=scores[version],public=float(row["evaluation"]["Public Score"]),
                            reference=row["submission_ref"],source=str(f.relative_to(ROOT))))
    table=pd.DataFrame(matches)
    if not table.version.is_unique:raise RuntimeError("Repeated releases need explicit reconciliation")
    table=table.sort_values("development",ascending=False)
    pairs=[]
    for a in range(len(table)):
        for b in range(a+1,len(table)):
            da=table.iloc[a].development-table.iloc[b].development
            db=table.iloc[a].public-table.iloc[b].public
            if 0<da<=.01 and db!=0:pairs.append(db>0)
    result=dict(versions=len(table),spearman=float(spearmanr(table.development,table.public).statistic),
                kendall_tau_b=float(kendalltau(table.development,table.public).statistic),
                within_one_point_non_tied_pairs=len(pairs),local_order_confirmed=int(sum(pairs)),
                median_local_minus_public_pp=float(np.median(table.development-table.public)*100),
                versions_by_development=table.to_dict(orient="records"),
                limitations="Exact version strings only, plus authenticated v96 OOF. Selected submissions, correlated pairs, different training sizes. No inference that any score is a data ceiling.")
    write(ROOT/"reports/common_error_followup/local_public_agreement.json",result)
    print({k:v for k,v in result.items() if k!="versions_by_development"})
    print(table[["version","development","public"]].to_string(index=False))


if __name__=="__main__":
    run()
