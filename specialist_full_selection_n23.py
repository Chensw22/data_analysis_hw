"""Verify archive parity, CARTE rerun sensitivity, paired gains and uniqueness before submission."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from kaggle_workflow import validate_submission
from specialist_full_route_n23 import (
    ACTIONS, ART, OUT, REPORT, ROOT, SEEDS, decision, digest, family_tags,
    predictions, routing_partitions, use_tree, write,
)


def local():
    row,y,_,groups,_=load()
    original_report=json.loads((ROOT/'reports/nov23_expert_dependence/routing_replay.json').read_text())
    scores, sensitivity={},{}
    for seed in SEEDS:
        b,p,fold=predictions(row,y,seed,ACTIONS)
        scores[str(seed)]={'base_accuracy':float(((b>=.5)==y).mean())}
        for rseed in [481,179]:
            r=original_report['policies'][f'original/real/{rseed}']
            pos=list(SEEDS).index(seed)
            h=float(r['accuracy'][pos])
            fixed,harmed=r['fixed'][pos],r['harmed'][pos]
            assert round(len(y)*(h-scores[str(seed)]['base_accuracy']))==fixed-harmed
            scores[str(seed)][str(rseed)]=dict(accuracy=h,fixed=fixed,harmed=harmed,net=fixed-harmed)
        if seed!=SEEDS[0]:
            continue
        with np.load(ART/f'nov23_carte/{seed}_trace.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            np.testing.assert_array_equal(z['fold'],fold)
            j=list(ACTIONS).index('native_carte')
            changed=int(((p[:,j]>=.5)!=(z['p']>=.5)).sum())
            p[:,j]=z['p']
        inputs=np.c_[b,p]
        for rseed in [481,179]:
            new=np.full(len(y),np.nan)
            for f in np.unique(fold):
                ix=np.flatnonzero(fold==f)
                trees=routing_partitions(inputs[ix])
                for t,q in GroupKFold(5,shuffle=True,random_state=rseed).split(ix,groups=groups[ix]):
                    new[ix[q]],_=use_tree(trees,inputs[ix],t,q,y[ix[t]])
            sensitivity[str(rseed)]=dict(changed_carte_votes=changed,
                **paired(decision(new,b)==y,(b>=.5)==y))
    return scores,sensitivity


def main():
    candidate=OUT/'submission.csv'
    check=validate_submission(candidate,ROOT/'data/raw/test.csv')
    rows=pd.read_csv(candidate,dtype={'PassengerId':str})
    receipts=[]
    matched={}
    for path in sorted((ROOT/'reports').glob('*/online_attempts/*.json')):
        r=json.loads(path.read_text())
        assert r.get('file_sha256')!=check['sha256'],path
        if r.get('evaluation',{}).get('Status')!='COMPLETE':
            continue
        ref=r['submission_ref']
        receipts.append(str(ref))
        src=ART/path.parent.parent.name/'submission.csv'
        if src.is_file() and digest(src)==r['file_sha256']:
            other=pd.read_csv(src,dtype={'PassengerId':str})
            np.testing.assert_array_equal(other.PassengerId,rows.PassengerId)
            matched[str(ref)]=dict(run=path.parent.parent.name,
                   differences=int((rows.Transported!=other.Transported).sum()))
    assert matched and min(v['differences'] for v in matched.values())>0
    modes,carte=local()
    parity={k:json.loads((OUT/f'parity_{tag}.json').read_text()) for k,tag in zip(ACTIONS,family_tags())}
    subjects={}
    for kind in ['mitra','v44']:
        p=ART/'proposal_crossfamily_v80'/kind/'complete.json'
        subjects[kind]=dict(metadata=digest(p),**json.loads(p.read_text()))
    s=dict(version='n23_all_ten_expert_posterior_vote',**check,
        prior_artifact_differences=matched,official_refs_with_local_receipts=len(set(receipts)),
        validated_prediction_matrices=len(matched),first_stage_parity=parity,
        test_differences_vs_v131=matched['58653601']['differences'],replays=modes,
        carte_new_train_sensitivity=carte,source_code={
            Path(p).name:digest(ROOT/p) for p in [
                'specialist_full_manifest_n23.py','specialist_full_route_n23.py','specialist_full_selection_n23.py',
                'broad_specialist_tree_average_n23.py','broad_booking_policy_n23.py',
                'broad_specialist_resolution_n23.py','projection_shapes_n23.py','pretrained_recheck_n23.py',
                'passenger_variational_gp_n23.py','carte_passengers_n23.py','manifest_likelihood_tree_n23.py',
                'class_analogue_experts_n23.py','grouped_field_factors_n23.py','bill_matched_register_n23.py',
                'spend_simplex_likelihood_n23.py','bayesian_manifest_forest_n23.py']},
        base_sources=subjects,export_json_sha=digest(REPORT/'export.json'),
        test_scores_sha=digest(OUT/'routed_test.npz'))
    write(REPORT/'selection.json',s)
    write(OUT/'selection.json',s)
    print('verified receipts',len(matched),'nearest',sorted([(v['differences'],r) for r,v in matched.items()])[:3],
          'CARTE sensitivity',carte,'archive scores',modes,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
