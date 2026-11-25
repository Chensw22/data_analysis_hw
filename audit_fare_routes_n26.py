"""Compare route-aware models to the frozen 98-change fare-deck rule on identical passengers."""
import json

import numpy as np

from export_fare_deck_n26 import combine
from fare_route_inference_n26 import DECK, OUT, REPORT
from solo_archive_recovery_n26 import ticker
from tabicl_consensus_review_n26 import OUT as REVIEW, sources, splits


def held_fare(panel, tag, with_routes=True, directory=OUT, architectures=('tree', 'neural')):
    ids = panel['ids']
    result = {}
    for fold in np.unique(panel['fold']):
        m = panel['fold'] == fold
        for arch in ('tree', 'neural'):
            files = [('deck', DECK)] + ([('route', directory)] if with_routes and arch in architectures else [])
            for stem, source in files:
                with np.load(source/f'{arch}_{tag}_{fold}.npz') as z:
                    np.testing.assert_array_equal(z['q'], ids[m])
                    for name in (['fine', 'raw'] if stem == 'deck' else ['exact','broad','displaced']):
                        col = result.setdefault(f'{arch}_{stem}_{name}', np.zeros(len(ids)))
                        col[m] = z[name]
    return result


def reviews(panels, group):
    batches = {}
    for key, _, dst, _, q in splits(panels, group):
        tag = key.split('_')[0] if key != 'historical_to_new' else key
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'], dst['ids'][q])
            flip = (z['identities_error'] > .5) & (z['identities_outcome'] > .5)
            batches.setdefault(tag, []).append((q, dst['anchor'][q] ^ flip))
    for tag, batch in batches.items():
        q, parent = (np.concatenate([b[i] for b in batch]) for i in range(2))
        assert len(np.unique(q)) == len(q)
        yield tag, q, parent


def run(swaps=False):
    if swaps:
        from receipt_charge_swaps_n26 import OUT as directory
        architectures = ('tree',)
        diagnostic = {'matched_charge_ranks': json.loads((directory/'diagnostic.json').read_text())}
        with np.load(directory/'scores.npz') as z:
            eligible = np.isfinite(z['exact']).any(axis=1)
        output = REPORT.with_name('receipt_charge_swaps.json')
    else:
        directory, architectures, output = OUT, ('tree','neural'), REPORT
        diagnostic = {'registered_destination': json.loads((OUT/'destination.json').read_text())}
        with np.load(OUT/'routes.npz') as z:
            eligible = z['valid']
    raw, y, g, panels = sources()
    predictions = {tag: held_fare(v, tag, directory=directory, architectures=architectures) for tag, v in panels.items()}
    report = diagnostic
    with np.load(DECK/'deck_estimates.npz') as z:
        deck_eligible = z['eligible']
    for tag, q, review in reviews(panels, g):
        base = 'historical' if tag == 'historical' else 'full'
        panel, all_p = panels[base], predictions[base]
        ids = panel['ids'][q]
        p = {k: v[q] for k, v in all_p.items()}
        fare = combine(review, p['neural_deck_raw'], p['neural_deck_fine'], deck_eligible[ids]) >= .5
        common = panel['common'][q]
        true = y[ids]
        wrong = fare != true
        masks = {'all': np.ones(len(q), bool), 'unanimous': common,
                 'remaining_original_unanimous_errors': common & (review != true) & wrong,
                 'previously_repaired_unanimous_errors': common & (review != true) & ~wrong}
        actions = {}
        for arch in architectures:
            deck = p[arch+'_deck_fine'] >= .5
            for key in ('exact','broad','displaced'):
                vote = p[arch+'_route_'+key] >= .5
                actions[arch+'_'+key] = (vote != fare) & (deck == fare) & eligible[ids]
        if 'neural' in architectures:
            actions['two_models'] = actions['tree_exact'] & actions['neural_exact']
        r = dict(rows=len(q), before=float((fare == true).mean()))
        for name, act in actions.items():
            delta = act * np.where(wrong, 1, -1)
            r[name] = {key: ticker(m, wrong, act) for key, m in masks.items()}
            r[name].update(after=float(r['before'] + delta.mean()),
                           net_by_fold=[int(delta[panel['fold'][q]==k].sum()) for k in np.unique(panel['fold'][q])])
        report[tag] = r
        print(tag, [(k,r[k]['all']['fixed'],r[k]['all']['harmed'],r[k]['unanimous']['fixed'],r[k]['unanimous']['harmed']) for k in actions], flush=True)
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--charge-swaps', action='store_true')
    run(parser.parse_args().charge_swaps)
