"""One bounded 100-step extension of the frozen run041 last-block procedure.

The only learning change is the maximum inner-selection/refit step count.
This wrapper never edits or resumes run041. Hitting 100 is unresolved budget
uncertainty, not permission to extend to 200 or alter layers/learning rate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import mitra_lastblock_candidate as original

ROOT = Path(__file__).resolve().parent
mitra = original.mitra  # Existing source-checked top5 exporter integration.
NAME = 'v76_Mitra_lastblock100'
CONTROL = original.CONTROL
REVIEW = 'reports/top5_lastblock50_review.json'
ORIGINAL_SETTINGS = copy.deepcopy(original.SETTINGS)
SETTINGS = copy.deepcopy(ORIGINAL_SETTINGS)
SETTINGS.update(max_steps=100, screen_budget_seconds=21600,
    stability_budget_seconds=64800, export_budget_seconds=10800,
    budgets=dict(screen=21600, validate=64800, export=10800))
OWN_SOURCES = list(original.OWN_SOURCES) + [
    'mitra_lastblock100_candidate.py', 'test_mitra_lastblock100_candidate.py', REVIEW]
_ORIGINAL_RECORD_PROTOCOL = original.record_protocol
_ORIGINAL_MAIN = original.main
_ORIGINAL_FIT_LASTBLOCK = original.fit_lastblock
_ORIGINAL_NAME = original.NAME
STOPPING_RULE = ('At most one extension to 100. If the best inner step is again '
    'at 100, record convergence/budget sufficiency as inconclusive; never '
    'automatically extend to 200. Completed accuracy remains a valid result of '
    'this bounded procedure, not proof of the whole adaptation family.')


def verify_basis():
    """Keep all original source files frozen; extension rationale is inner-only."""
    history = ROOT/'artifacts/run_041/protocol.json'
    saved = json.loads(history.read_text())
    for relative, expected in saved['sources'].items():
        if original.mitra.sha256(ROOT/relative) != expected:
            raise RuntimeError(f'Frozen run041 source changed: {relative}')
    review = json.loads((ROOT/REVIEW).read_text())
    if review['finite_followup']['lastblock_extension_spec']['max_steps'] != 100:
        raise RuntimeError('The reviewed one-extension limit changed.')
    if sum(row['selected_steps'] == 50 for row in review['curves']) != 2:
        raise RuntimeError('Expected two inner folds at the declared 50-step cap.')
    return dict(original_run='run_041', original_candidate='v75_Mitra_lastblock50',
        original_protocol_sha256=original.mitra.sha256(history),
        review=REVIEW, review_sha256=original.mitra.sha256(ROOT/REVIEW),
        reason='Two inner selections at50 plus late inner-logloss improvement; not outer score optimization.',
        only_learning_change=dict(max_steps_before=50, max_steps_after=100),
        stopping_rule=STOPPING_RULE, resume_previous_fit=False)


def configure():
    if _ORIGINAL_NAME != 'v75_Mitra_lastblock50':
        raise RuntimeError('Import this wrapper in a fresh process before configuring another version.')
    if original.NAME == NAME:
        if original.SETTINGS != SETTINGS or original.OWN_SOURCES != OWN_SOURCES:
            raise RuntimeError('Configured 100-step recipe changed in memory.')
        original.configure()
        return
    if original.iterate.fit is not original.BASE_FIT or original.iterate.evaluate is not original.BASE_EVALUATE:
        raise RuntimeError('Do not repurpose an already configured run041 process.')
    verify_basis()
    original.NAME = NAME
    original.SETTINGS = copy.deepcopy(SETTINGS)
    original.OWN_SOURCES = list(OWN_SOURCES)
    original.record_protocol = record_protocol
    original.fit_lastblock = annotated_fit_lastblock
    original.configure()


def record_protocol(out):
    configure()
    basis = verify_basis()
    _ORIGINAL_RECORD_PROTOCOL(out)
    path = Path(out)/'protocol.json'
    record = json.loads(path.read_text())
    if 'bounded_lastblock_extension' in record:
        if record['bounded_lastblock_extension'] != basis:
            raise RuntimeError('Frozen extension basis changed.')
        return
    record['lastblock_policy']['selection'] = (
        'inner80/20, steps0..100 minbinaryLL earliestties; freshfullrefit')
    record['bounded_lastblock_extension'] = basis
    original.write_json(path, record)


def configure_runtime(out, stage='screen'):
    configure()
    original.configure_runtime(out, stage=stage)


def assert_frozen(out):
    configure()
    original.assert_frozen(out)
    record = json.loads((Path(out)/'protocol.json').read_text())
    if record.get('bounded_lastblock_extension') != verify_basis():
        raise RuntimeError('Frozen bounded-extension declaration differs.')
    if 'steps0..100' not in record['lastblock_policy']['selection']:
        raise RuntimeError('Unexpected inner selection range.')


def annotated_fit_lastblock(*args, **kwargs):
    artifact = _ORIGINAL_FIT_LASTBLOCK(*args, **kwargs)
    # Metadata only: every entry path (screen, generic stability and export)
    # runs the same original learner and gets the same bounded-stop annotation.
    at_cap = bool(artifact.metadata.get('selection_at_cap'))
    artifact.metadata.update(bounded_extension_max_steps=100,
        convergence_assessment='inconclusive_at_cap' if at_cap else 'inner_selection_below_cap',
        automatic_further_extension_allowed=False)
    log = kwargs.get('log_path', args[5] if len(args) > 5 else None)
    original.write_json(log, artifact.metadata)
    return artifact


def fit(name, matrices, train_idx, y, predict_idx, seed):
    configure()
    return original.fit(name, matrices, train_idx, y, predict_idx, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen', 'stability', 'export'])
    parser.add_argument('--run', default='run_042')
    args = parser.parse_args()
    configure()
    if args.stage == 'stability':
        import top5_validate
        top5_validate.validate('mitra_lastblock100_candidate', args.run)
        return
    # Keep the audited original orchestration, guards, artifact formats and
    # timeout cleanup; explicitly pass the new run instead of its old default.
    previous = sys.argv
    try:
        sys.argv = [__file__, args.stage, '--run', args.run]
        _ORIGINAL_MAIN()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    sys.modules['mitra_lastblock100_candidate'] = sys.modules[__name__]
    main()
