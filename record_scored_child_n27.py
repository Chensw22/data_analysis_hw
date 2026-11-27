"""Record a submitted descendant after verifying its parent and file against both official receipts."""
import argparse
import json
from pathlib import Path

from await_half_roster_slot_n23 import snapshot
from await_service_pair_followup_n23 import persist_score
from await_service_pair_slot_n23 import available
from kaggle_workflow import sha256

ROOT = Path(__file__).resolve().parent


def run(name, parent, model, source):
    folder = ROOT / 'reports' / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    receipt, = (folder / 'online_attempts').glob('*.json')
    attempt = json.loads(receipt.read_text())
    base = ROOT / 'reports' / parent
    first = json.loads((base / 'followup_result.json').read_text())
    predecessor = json.loads((base / 'manifest.json').read_text())
    assert manifest['parents'][parent] == predecessor['sha256']
    assert attempt['evaluation']['Status'] == 'COMPLETE' and attempt['file_sha256'] == manifest['sha256']
    assert sha256(ROOT / 'artifacts' / name / model) == manifest['model_sha256']
    api, limits = available()
    _, third = snapshot(api)
    print('Official balance and third place', limits, third, flush=True)
    persist_score(name, manifest, receipt, model, manifest['sha256'], source, attempt['commit'], manifest['parents'], third, first)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('name', 'parent', 'model', 'source'):
        parser.add_argument(key)
    args = parser.parse_args()
    run(args.name, args.parent, args.model, args.source)
