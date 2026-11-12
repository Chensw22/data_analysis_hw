"""Practical one-percentage-point bands, not significance/equivalence tests."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    versions = {}
    for path in sorted((ROOT/'reports').glob('run_*/screen_results.csv')):
        for row in csv.DictReader(path.open()):
            name, accuracy = row['version'], float(row['accuracy'])
            if name in versions:
                if abs(versions[name]['accuracy']-accuracy) > 1e-12:
                    raise ValueError(f'Conflicting development scores for {name}; review before grading.')
                versions[name]['source_runs'].append(path.parent.name)
            else:
                versions[name] = dict(version=name, accuracy=accuracy, source_runs=[path.parent.name])
    best = max(row['accuracy'] for row in versions.values())
    rows = []
    for row in sorted(versions.values(), key=lambda r: (-r['accuracy'], r['version'])):
        gap = (best-row['accuracy'])*100
        tier = 'A' if gap <= 1+1e-12 else 'B' if gap <= 2+1e-12 else 'C'
        rows.append(dict(**row, gap_from_best_pp=gap, tier=tier,
            near_band_boundary=bool(min(abs(gap-1), abs(gap-2)) < .15),
            stable_superiority_established=False))
    output = ROOT/'reports/method_tiers.csv'
    with output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({**row, 'source_runs': ';'.join(row['source_runs'])} for row in rows)
    policy = dict(reference_accuracy=best, unit='percentage_points',
        bands={'A':'gap from reference <=1pp', 'B':'1pp < gap <=2pp', 'C':'gap >2pp'},
        reference_policy='Anchor all bands to the same current best score. Pairwise distance<=1pp is not transitive.',
        statistical_claim='None. Practical screening bands do not prove statistical equivalence or stable inferiority; adjacent-band models can differ by much less than1pp.',
        scope='Completed fixed development3fold recipes only, not entire model families. Controls and blends are retained as named recipes. Repeated identical versions are deduplicated.',
        stability='No current recipe is established as a stable winner. v34 fresh-seed gains are small and GroupCV reverses them; leading neural recipes have no matching repeated-seed/group confirmation.',
        future_stability_review='Treat <=1pp differences as the same practical performance level for a direct comparison. Claim a clearly stronger recipe only after repeated paired splits give consistent material gains and group validation does not reverse the direction; one public leaderboard result is insufficient.',
        online_evidence='All eight A-tier representatives have completed account submissions; see reports/a_tier_batch_20261112/results.json. Mitra-v2 remains highest on the saved2026-11-12 public test, but this is not a stable-superiority certificate.',
        s_tier=[], recipe_count=len(rows), counts={tier:sum(r['tier']==tier for r in rows) for tier in 'ABC'})
    (ROOT/'reports/method_tiers.json').write_text(json.dumps(policy, indent=2)+'\n')
    print(json.dumps(policy['counts']), f'{len(rows)} completed recipes; reference={100*best:.5f}%')


if __name__ == '__main__':
    main()
