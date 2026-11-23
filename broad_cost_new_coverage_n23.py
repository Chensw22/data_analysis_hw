"""Distinguish genuinely new corrections from those already made by ordinary score thresholds."""
import numpy as np

from broad_class_cost_path_n23 import OUT, REPORT
from broad_partition_evidence_n23 import SEEDS, historical, write


def main():
    row, y, masks = historical()
    families = {'cost': ('negative_weight', 'positive_weight'),
                'threshold': ('negative_threshold_control', 'positive_threshold_control')}
    panels = {name: [] for name in families}
    for seed in SEEDS:
        with np.load(OUT / f'{seed}.npz') as data:
            np.testing.assert_array_equal(data['row'], row)
            np.testing.assert_array_equal(data['y'], y)
            for name, keys in families.items():
                panels[name].append(np.column_stack([data[key] == y for key in keys]))
    # A passenger counts only if the same specialist got that passenger right on all three splits.
    stable = {name: np.stack(scores).all(0).any(1) for name, scores in panels.items()}
    report = {}
    for key in ('ever', 'repeat', 'unresolved', 'never'):
        selected = masks[key]
        cost, threshold = stable['cost'], stable['threshold']
        cells = {
            'both': cost & threshold, 'only_trained_cost': cost & ~threshold,
            'only_threshold': threshold & ~cost, 'neither': ~cost & ~threshold,
        }
        report[key] = {name: dict(total=int((selected & mask).sum()),
                                  transported=int((selected & mask & y).sum()))
                       for name, mask in cells.items()}
        assert sum(x['total'] for x in report[key].values()) == int(selected.sum())
    write(REPORT / 'new_coverage.json', report)
    print(report)


if __name__ == '__main__':
    main()
