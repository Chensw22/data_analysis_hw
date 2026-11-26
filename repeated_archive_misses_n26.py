"""Measure simultaneous and repeatable coverage of unanimous errors by archived booking-held models."""
import json

import numpy as np

from archive_transport_columns_n26 import read_probability_columns
from full_roster_booking_validation_n23 import ROOT, data

ART = ROOT / 'artifacts'
OUT = ART / 'nov26_repeated_archive_misses'
REPORT = ROOT / 'reports/nov23_matched_learning_curve/repeated_archive_misses.json'


def cohort(raw, y, fold):
    with np.load(ART / 'nov26_retained_order_mechanisms/aligned.npz') as archive:
        for key, expected in [('passenger', raw.PassengerId), ('y', y), ('fold', fold)]:
            np.testing.assert_array_equal(archive[key], expected)
        base = archive['reference161']
        blind = (archive['eligible'] & ((archive['original_panel'] >= .5) == base[:, None]).all(1)
                 & ((archive['pairs'] >= .5) == base[:, None, None]).all(axis=(1, 2)))
        assert blind.sum() == 3861 and (blind & (base != y)).sum() == 266
        return base, blind, archive['known'], archive['fresh']


def collect(folder, tag, rows, fold, groups, y, key):
    blocks = []
    labels = None
    for k in np.unique(fold):
        hold = fold == k
        q = rows[hold]
        t = rows[~hold & ~np.isin(groups[rows], groups[q])]
        if folder.name == 'nov26_joint_party_tabicl':
            # The joint model only fits complete training bookings because its target uses every member.
            _, code, sizes = np.unique(groups, return_inverse=True, return_counts=True)
            available = np.bincount(code[t], minlength=len(sizes))
            t = t[available[code[t]] == sizes[code[t]]]
        with np.load(folder / f'{tag}_{k}.npz') as archive:
            def field(*choices):
                return archive[next(c for c in choices if c in archive)]
            np.testing.assert_array_equal(field('t', 'train'), t)
            np.testing.assert_array_equal(field('q', 'query'), q)
            np.testing.assert_array_equal(field('truth', 'labels', 'train_labels', 'train_y', 'y'), y[t])
            if key is None:
                cols = read_probability_columns(folder, archive)
                names = list(cols)
                p = np.column_stack(list(cols.values()))
            else:
                p = archive[key]
                names = [str(j) for j in range(p.shape[1])]
            if labels is None:
                labels = names
            else:
                assert labels == names
            assert p.ndim == 2 and len(p) == len(q) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
            blocks.append((np.flatnonzero(hold), p))
    scores = np.empty((len(rows), blocks[0][1].shape[1]))
    for ix, p in blocks:
        scores[ix] = p
    return scores, labels


def count(edits, mask, wrong):
    return dict(fixed=int((edits & mask & wrong).sum()), harmed=int((edits & mask & ~wrong).sum()))


def run():
    raw, y, g, f = data()
    base, blind, known, fresh = cohort(raw, y, f)
    with np.load(ART / 'nov25_historical_nine/historical.npz') as z:
        rows, oldfold = z['rows'], z['fold']
        np.testing.assert_array_equal(z['passenger'], raw.PassengerId.iloc[rows])
        np.testing.assert_array_equal(z['y'], y[rows])
    pooled, old, names, families = [], [], [], []
    skipped = {}
    for first in sorted(ART.glob('*/full_0.npz')):
        folder = first.parent
        if not (folder / 'historical_0.npz').exists():
            continue
        with np.load(first) as z:
            key = next((k for k in ('p', 'prob', 'scores') if k in z), None)
            if key is None and not read_probability_columns(folder, z):
                skipped[folder.name] = 'no passenger probability matrix'
                continue
            if not np.array_equal(z['query' if 'query' in z else 'q'], np.flatnonzero(f == 0)):
                skipped[folder.name] = 'partial roster'
                continue
        full, fnames = collect(folder, 'full', np.arange(len(y)), f, g, y, key)
        past, hnames = collect(folder, 'historical', rows, oldfold, g, y, key)
        assert full.shape[1] == past.shape[1] and fnames == hnames
        families.extend([folder.name.removesuffix('_repeat')] * full.shape[1])
        names.extend([f'{folder.name}/{j}' for j in fnames])
        pooled.append(full)
        old.append(past)
    full, historical = np.column_stack(pooled), np.column_stack(old)
    names, families = np.array(names), np.array(families)
    wrong = base != y
    # A probability of exactly .5 is an abstention (some peer methods explicitly emit it without a peer).
    signed = (full - .5) * np.where(base, -1., 1.)[:, None]
    historical_signed = (historical - .5) * np.where(base[rows], -1., 1.)[:, None]
    changed = signed > 0
    repeated = changed[rows] & (historical_signed > 0)
    eligible = dict(all=known, blind=blind, fresh_blind=blind & fresh,
                    awake_blind=blind & raw.CryoSleep.eq(False).to_numpy(), asleep_blind=blind & raw.CryoSleep.eq(True).to_numpy())
    family = np.unique(families)

    def analyze(matrix, subset):
        diff = np.column_stack([matrix[:, families == k].any(1) for k in family])
        votes = diff.sum(1)
        results = {}
        for title, mask in eligible.items():
            mask = mask[subset]
            error = wrong[subset]
            results[title] = dict(rows=int(mask.sum()), errors=int((mask & error).sum()),
                any=count(votes > 0, mask, error),
                wrong_by_family_count=np.bincount(votes[mask & error], minlength=len(family)+1).tolist(),
                right_by_family_count=np.bincount(votes[mask & ~error], minlength=len(family)+1).tolist(),
                families={k:count(col, mask, error) for k, col in zip(family, diff.T, strict=True)},
                configurations={k:count(col, mask, error) for k, col in zip(names, matrix.T, strict=True)})
        return results

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'aligned.npz', passenger=raw.PassengerId.to_numpy(str), y=y, fold=f, rows=rows,
                        oldfold=oldfold, names=names, families=families, full=full, historical=historical)
    result = dict(columns=len(names), experiment_groups=len(family), omitted=skipped,
                  full=analyze(changed, np.arange(len(y))), full_matched=analyze(changed[rows], rows),
                  both_matched=analyze(repeated, rows),
                  full_margin=analyze(signed > .05, np.arange(len(y))),
                  repeated_margin=analyze((signed[rows] > .05) & (historical_signed > .05), rows))
    # Profiles include matching originally-correct people, so a recurring error phenotype can be evaluated.
    positive = raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).any(axis=1).to_numpy()
    single = ~raw.PassengerId.str.split('_').str[0].duplicated(keep=False).to_numpy()
    states = dict(awake_paid=raw.CryoSleep.eq(False).to_numpy() & positive,
                  asleep_unpaid=raw.CryoSleep.eq(True).to_numpy() & ~positive,
                  solo=single, shared_order=~single, missing_field=raw.isna().any(axis=1).to_numpy())
    result['never_challenged_profiles'] = {}
    for name, mask in {'all': np.ones(len(y),bool), **states}.items():
        match = blind & mask & ~changed.any(1)
        result['never_challenged_profiles'][name] = dict(wrong=int((match & wrong).sum()),
                                                          correct=int((match & ~wrong).sum()))
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(len(names), len(family), [(k, result[k]['blind']['errors'], result[k]['blind']['any'])
                                    for k in ('full', 'full_matched', 'both_matched')])


if __name__ == '__main__':
    run()
