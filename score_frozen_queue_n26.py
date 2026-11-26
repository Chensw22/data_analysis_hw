"""Compare the pre-upload forecast and paired local changes with official scores in the refreshed batch."""
import json
from datetime import date, datetime

import numpy as np
from scipy.stats import spearmanr

from full_roster_booking_validation_n23 import ROOT

FOLDER = ROOT / 'reports/nov23_matched_learning_curve'
RESERVE = 'proposal_n26_historical_veto'
DAY = date(2026, 11, 26)


def actual(name, digest):
    manifest = json.loads((ROOT / 'reports' / name / 'manifest.json').read_text())
    assert manifest['sha256'] == digest
    found = []
    for path in (ROOT / 'reports' / name / 'online_attempts').glob('*.json'):
        record = json.loads(path.read_text())
        result = record.get('evaluation', {})
        if record['file_sha256'] == digest and result.get('Status') == 'COMPLETE':
            submitted = datetime.fromisoformat(str(result['Submission Date']).replace('Z', '+00:00'))
            if submitted.date() == DAY:
                assert str(result['Submission Ref']) == str(record['submission_ref'])
                found.append((float(result['Public Score']), str(record['submission_ref'])))
    assert len(found) <= 1
    return found[0] if found else None


def run():
    forecast = json.loads((FOLDER / 'frozen_queue_forecast.json').read_text())
    entries, statuses, references = {}, {}, set()
    incumbent = forecast['starting_public_score']
    proposals = list(forecast['versions'].items())
    manifest = json.loads((ROOT / 'reports' / RESERVE / 'manifest.json').read_text())
    proposals.append(('reserve', dict(proposal=RESERVE, sha256=manifest['sha256'])))
    for short, item in proposals:
        name, dig = item['proposal'], item['sha256']
        state = json.loads((ROOT / 'artifacts' / name / 'queue_status.json').read_text())
        assert state['sha256'] == dig
        statuses[short] = state['phase']
        score = actual(name, dig)
        if score is None:
            continue
        value, ref = score
        assert ref not in references
        references.add(ref)
        entry = dict(proposal=name, public=value, public_gain_pp=100*(value-incumbent), ref=ref)
        if short != 'reserve':
            delta = item['local_fixed'] - item['local_harmed']
            entry.update(local_accuracy=item['local_accuracy'], local_net=delta,
                         local_gain_pp=100*delta/item['local_rows'], predicted_public=item['conditional_public_score'])
        entries[short] = entry
    pairs = [v for v in entries.values() if 'local_accuracy' in v]
    summary = dict(official_count=len(entries), comparable=len(pairs), incumbent_public=incumbent,
                   public_improved=sum(v['public'] > incumbent for v in entries.values()), phases=statuses, scores=entries)
    if pairs:
        y = np.array([v['public'] for v in pairs])
        preds = np.array([v['predicted_public'] for v in pairs])
        x = np.array([v['local_accuracy'] for v in pairs])
        summary['mean_forecast_error_pp'] = float(100*(preds-y).mean())
        summary['mae_forecast_pp'] = float(100*abs(preds-y).mean())
        summary['mae_unchanged_incumbent_pp'] = float(100*abs(incumbent-y).mean())
        if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
            summary['local_public_spearman'] = float(spearmanr(x,y).statistic)
    if len(entries) == 10:
        (FOLDER / 'nov26_official_queue_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    run()
