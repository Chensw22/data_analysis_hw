"""Describe confidence on authenticated historical shared errors; never fit a model."""
import json
from pathlib import Path

import numpy as np

import error_overlap_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parent


def main():
    diagnostic.common.require_committed(['shared_error_confidence.py'])
    inputs, _, _, probabilities, evidence = diagnostic.load_authenticated()
    y = inputs['y']
    p = np.column_stack([probabilities[name] for name in diagnostic.FAMILIES])
    wrong = (p >= .5) != y[:, None]
    count = wrong.sum(axis=1)
    old = json.loads((diagnostic.OUT/'error_overlap.json').read_text())
    if {str(i): int((count == i).sum()) for i in range(6)} != old['family_error_count_histogram']:
        raise RuntimeError('Original shared-error membership changed.')
    confidence = np.maximum(p, 1-p)
    v81 = confidence[:, diagnostic.FAMILIES.index('v81')]
    bins = [.5, .6, .8, .95, np.nextafter(1., np.inf)]
    groups = {}
    for name, mask in [('all_five_wrong', count == 5), ('all_five_correct', count == 0),
                       ('families_disagree', (count > 0) & (count < 5))]:
        n = int(mask.sum())
        groups[name] = dict(
            rows=n,
            true_positive_labels=int(y[mask].sum()),
            v81_confidence_bins_0_5_0_6_0_8_0_95_1=np.histogram(v81[mask], bins=bins)[0].tolist(),
            v81_confidence_median=float(np.median(v81[mask])),
            all_five_confidence_at_least_0_8=int(np.sum(np.all(confidence[mask] >= .8, axis=1))),
            all_five_confidence_at_least_0_95=int(np.sum(np.all(confidence[mask] >= .95, axis=1))),
            per_model_median_confidence={model: float(np.median(confidence[mask, i]))
                                         for i, model in enumerate(diagnostic.FAMILIES)},
        )
        if sum(groups[name]['v81_confidence_bins_0_5_0_6_0_8_0_95_1']) != n:
            raise RuntimeError('Confidence bins do not cover all rows.')
    result = dict(
        kind='additional_descriptive_shared_error_confidence_not_model_selection',
        rows=len(y), families=diagnostic.FAMILIES, groups=groups,
        bin_intervals='[.5,.6),[.6,.8),[.8,.95),[.95,1]',
        interpretation=[
            'Confidence is a model estimate, not verified label correctness or irreducible uncertainty.',
            'Confident shared errors can reflect common bias, absent features or label issues; these counts do not distinguish their causes.',
            'These reused development predictions do not authorize relabeling, dropping hard rows, subgroup routing or threshold tuning.',
            'No model fit, holdout label access, new candidate score or online upload.',
        ],
        evidence_sha256=evidence,
        source_sha256=diagnostic.audit.file_hash(Path(__file__)),
        original_report_sha256=diagnostic.audit.file_hash(diagnostic.OUT/'error_overlap.json'),
    )
    diagnostic.audit.json_write(diagnostic.OUT/'shared_error_confidence.json', result)
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
