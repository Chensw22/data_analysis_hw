"""Evaluate aggregate local-to-public forecasts on later batches, using earlier scored batches only."""
from itertools import combinations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_recent_family_transfer_n23 import BASE, RELEASES, confirmed, file_votes, history
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ROOT, write
from export_pending_family_support_n23 import NAME as SUPPORT
from pending_family_coverage_n23 import PAIRS, development_votes, read, rules
from released_correction_audit_n23 import REPORT as OLD, development, release

REPORT = ROOT/'reports/nov23_forward_release_transfer'


def measurements(name, fixed, harmed, changed, n_local, n_test):
    assert fixed+harmed > 0 and changed > 0
    return dict(name=name, fixed=int(fixed), harmed=int(harmed), test_changes=int(changed),
                local_gain_pp=100*(fixed-harmed)/n_local,
                edit_pp=100*changed/n_test,
                scaled_gain_pp=100*changed/n_test*(fixed-harmed)/(fixed+harmed))


def inputs(ids):
    official = read(ROOT/'reports/nov23_flip_capacity/primary_scores.json')['public_scores']
    y, votes, _ = development()
    records = {a: [] for a in (108, 128, 131)}
    older = {n: release(n, ids, official) for n in votes}
    for entry in read(OLD/'results.json')['rows']:
        n, a = entry['version'], entry['anchor']
        hit, anchor = votes[n] == y, votes[a] == y
        fixed, harmed = (hit & ~anchor).sum(), (~hit & anchor).sum()
        score, current, receipt, _ = older[n]
        reference, other, _, _ = older[a]
        changes = (current != other).sum()
        assert (fixed, harmed, changes) == (entry['fixed'], entry['harmed'], entry['test_changes'])
        m = measurements('v'+str(n), fixed, harmed, changes, len(y), len(ids))
        records[a].append(dict(**m, public_gain_pp=100*(score-reference), receipt=receipt))
    for r in history(ids):
        name = r['name']
        _, path, key = RELEASES[name]
        replay = read(ROOT/'reports'/path)['plans']['v131'][key]
        m = measurements(name, replay['fixed'], replay['harmed'], r['test_changes'], len(y), len(ids))
        records[131].append(dict(**m, public_gain_pp=r['public_gain_pp'], receipt=r['receipt']))
    return records


def predict(earlier, future):
    total_edits = sum(x['edit_pp'] for x in earlier)
    realized = sum(x['public_gain_pp'] for x in earlier)/total_edits
    bias = sum(x['public_gain_pp']-x['scaled_gain_pp'] for x in earlier)/total_edits
    return {r['name']: dict(no_change=0., local_net=r['local_gain_pp'], scaled_net=r['scaled_gain_pp'],
                            historical_edit_rate=r['edit_pp']*realized,
                            adjusted_net=r['scaled_gain_pp']+r['edit_pp']*bias) for r in future}


def evaluate(earlier, later):
    assert max(read(ROOT/r['receipt'])['last_checked_at_utc'] for r in earlier) < min(
        read(ROOT/r['receipt'])['started_at_utc'] for r in later)
    predicted = predict(earlier, later)
    metrics = {}
    actual = np.array([r['public_gain_pp'] for r in later])
    for method in next(iter(predicted.values())):
        estimate = np.array([predicted[r['name']][method] for r in later])
        concordance = [np.sign(actual[i]-actual[j])*np.sign(estimate[i]-estimate[j])
                       for i,j in combinations(range(len(later)), 2) if abs(actual[i]-actual[j]) > 1e-8]
        winners = np.flatnonzero(np.isclose(estimate, estimate.max(), rtol=0, atol=1e-10))
        metrics[method] = dict(mae_pp=float(np.abs(estimate-actual).mean()),
                               pairs_correct=int(np.sum(np.array(concordance)>0)), pairs_total=len(concordance),
                               pairs_tied=int(np.sum(np.array(concordance)==0)),
                               choices=[later[i]['name'] for i in winners],
                               mean_selected_gain_pp=float(actual[winners].mean()), best_available_pp=float(actual.max()))
    return dict(n_training=len(earlier), predictions=predicted, metrics=metrics)


def pending(ids):
    row, y, frame, groups, _ = load()
    weight = read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    ref, votes = development_votes(frame, groups, row, y, 'v131', weight)
    votes[SUPPORT] = ref ^ rules(ref, votes)[0]['queued_or_two']
    official = read(ROOT/'reports/nov23_flip_capacity/primary_scores.json')['public_scores']
    _, incumbent, _ = confirmed(BASE, ids, official)
    rows = []
    for name, pred in votes.items():
        hit, anchor = pred == y, ref == y
        test, _ = file_votes(name, ids)
        rows.append(measurements(name, (hit & ~anchor).sum(), (~hit & anchor).sum(),
                                 (incumbent != test).sum(), len(y), len(ids)))
    return rows


def main():
    ids = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId']).PassengerId
    batches = inputs(ids)
    prior = batches[108]
    results = {}
    for anchor in (128, 131):
        results[anchor] = evaluate(prior, batches[anchor])
        prior = prior+batches[anchor]
    options = pending(ids)
    write(REPORT/'results.json', dict(batches=batches, later_batches=results, pending=options,
                                     pending_forecasts=predict(prior, options)))
    for anchor, data in results.items():
        print(anchor, {k: (round(v['mae_pp'],3), v['pairs_correct'], v['pairs_total']) for k,v in data['metrics'].items()})
    print([(r['name'], round(v['adjusted_net'],3)) for r,v in zip(options,predict(prior, options).values())])


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
