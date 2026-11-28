"""Pair today's submitted files with the frozen direction model on both evaluations."""
from decimal import Decimal
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT/'reports'
STUDIES = REPORTS/'nov23_matched_learning_curve'
BASE = 'proposal_n27_directional_foundation'


def read(path):
    return json.loads(path.read_text())


def run():
    sources = [
        ('proposal_n27_conditional_party_allocation', STUDIES/'conditional_party_allocation.json',
         [(s, 'rules', 'supported_increment', 'scopes', 'all') for s in ['random','ordered']]),
        ('proposal_n27_complete_receipt_average', None,
         [('results', k, 'scopes', 'complete') for k in ['random_average_random_calibrated','ordered_average_ordered_calibrated']]),
        ('proposal_n28_divided_manifest_axes', None, [(s,'all') for s in ['random','ordered']]),
        ('proposal_n28_unanimous_architecture_opposition', STUDIES/'standalone_architecture_opposition_n28.json',
         [(s,'all') for s in ['random','contiguous']]),
        ('proposal_n28_deletion_confirmed_routes', None, [('results',s,'base','all') for s in ['random','ordered']]),
        ('proposal_n28_unanimous_payment_dispatch', None, [('results',s,BASE,'all') for s in ['random','ordered']]),
        ('proposal_n28_repeated_block_response', STUDIES/'matched_block_response_export_n28.json', [('cohorts','all')]),
    ]
    for name in ['proposal_n28_joint_foundation_raw','proposal_n28_tied_raw_crosscheck']:
        sources.append((name,None,[('evidence',s,'all') for s in ['random','ordered']]))
    sources.append(('proposal_n28_profile_and_bill_review',None,
                    [('evidence',s,'direction','all') for s in ['random','ordered']]))
    def submission(name):
        return pd.read_csv(ROOT/'artifacts'/name/'submission.csv').set_index('PassengerId').Transported
    base = submission(BASE)
    reference = Decimal(str(read(REPORTS/BASE/'followup_result.json')['score']))
    rows = []
    for name, study, keys in sources:
        folder = REPORTS/name
        study = study or folder/'selection.json'
        raw = read(study)
        nets = []
        for fields in keys:
            pair = raw
            for field in fields:
                pair = pair[field]
            nets.append(pair['fixed']-pair['harmed'])
        pred = submission(name)
        assert pred.index.equals(base.index)
        result = read(folder/'followup_result.json')
        score = Decimal(str(result['score']))
        rows.append(dict(name=name, source=str(study.relative_to(ROOT)), local_net=nets, changed=int((pred!=base).sum()),
                         public=float(score), public_delta_pp=float(100*(score-reference)), ref=result['ref']))
    output = dict(reference=BASE, reference_public=float(reference), known_rows=8532, submissions=rows,
                  above=sum(r['public_delta_pp']>0 for r in rows), tied=sum(r['public_delta_pp']==0 for r in rows),
                  below=sum(r['public_delta_pp']<0 for r in rows))
    (STUDIES/'nov28_same_parent_local_public.json').write_text(json.dumps(output,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    run()
