"""Prepare a reviewable local CSV; does not authorize or perform online submission.

The user asked whether the current best is worth submitting. The original gate
remains failed; an explicit one-time exception is required before uploading.
"""
import hashlib
import json
import shutil
import subprocess
import joblib
import numpy as np
import pandas as pd
import contrast_features
import iterate
from kaggle_workflow import validate_submission


def main():
    source=iterate.ROOT/'artifacts/run_012'
    iterate.assert_frozen(source)
    audit=json.loads((iterate.ROOT/'artifacts/audit_run012/audit.json').read_text())
    out=iterate.ROOT/'artifacts/proposal_run012'
    if out.exists():
        raise RuntimeError('Proposal already exists; do not overwrite.')
    out.mkdir()
    train,test,matrices,_=iterate.dataset()
    test_idx=np.arange(len(train),len(train)+len(test))
    probability,model=iterate.fit('v34_contrasts_spatial',matrices,np.arange(len(train)),
        train.Transported.astype(int).to_numpy(),test_idx,iterate.SEEDS[0])
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    joblib.dump(model,out/'model.joblib')
    reloaded=joblib.load(out/'model.joblib')
    np.testing.assert_allclose(probability,reloaded.predict_proba(matrices[False][0].iloc[test_idx])[:,1],rtol=0,atol=1e-12)
    pd.DataFrame({'PassengerId':test.PassengerId,'Transported':probability>=.5}).to_csv(out/'submission.csv',index=False)
    pd.DataFrame({'PassengerId':test.PassengerId,'probability':probability}).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(out/'submission.csv',iterate.ROOT/'data/raw/test.csv')
    selection=dict(version='v34_contrasts_spatial',components=['v34_contrasts_spatial'],source_run='run_012',
                   stage='proposal_awaiting_explicit_user_exception',screen_accuracy=.8192407247627265)
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    for name in ['contrast_features.py','spatial_expert.py','iterate.py','experiment.py','prepare_submission_proposal.py']:
        shutil.copy2(iterate.ROOT/name,out/name)
    manifest=dict(**check,selection=selection,fit_rows=len(train),model_reload_verified=True,
        prepared_from_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=iterate.ROOT,text=True).strip(),
        sources={n:hashlib.sha256((iterate.ROOT/n).read_bytes()).hexdigest() for n in ['contrast_features.py','spatial_expert.py','iterate.py','experiment.py','prepare_submission_proposal.py']},
        original_83_percent_gate_passed=False,stability_gate_passed=audit['stable_relative_improvement'],
        submission_authorized=False,online_submitted=False,
        authorization_required='Explicit one-time user exception to >83% and stability gates, and approval to upload this exact CSV.',
        audit=audit)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=iterate.ROOT/'reports/proposal_run012';report.mkdir(exist_ok=False)
    for name in ['manifest.json','selection.json','prepare_submission_proposal.py']:
        shutil.copy2(out/name,report/name)
    print(json.dumps(check),flush=True)


if __name__=='__main__':
    main()
