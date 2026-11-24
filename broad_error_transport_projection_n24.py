"""Fit a sum of learned one-dimensional response curves across the full passenger roster."""
import json

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from full_roster_champion_overlap_n24 import aligned
from full_roster_class_neighborhood_mass_n24 import protocol
from full_roster_conditional_pool_n23 import ROOT, banks, data, latest_reference
from full_roster_disagreement_calibration_n24 import stats
from full_roster_mistake_influence_n24 import design

MODES = ('linear', 'fixed_curves', 'learned_curves')
TAG = 'broad_error_transport_projection'


class RidgeCurves(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.mode = mode
        self.linear = nn.Linear(width, 1)
        self.direction = nn.Parameter(torch.randn(12, width), requires_grad=(mode == 'learned_curves'))
        self.height = nn.Parameter(torch.zeros(12, 9))
        self.register_buffer('knots', torch.linspace(-2., 2., 9))

    def forward(self, x):
        score = self.linear(x).flatten()
        if self.mode == 'linear':
            return score
        unit = nn.functional.normalize(self.direction, dim=1)
        projected = x @ unit.T
        # Hat basis makes each one-dimensional curve linear between adjacent knots.
        hats = (1 - 2 * (projected.clamp(-2., 2.)[:, :, None] - self.knots).abs()).clamp_min(0)
        return score + (hats * self.height).sum((1, 2))


def predict(x, y, groups, train, query):
    assert not np.isin(groups[train], groups[query]).any()
    centered = (x - x[train].mean(0)) / x[train].std(0).clip(.15)
    z = torch.tensor(centered.clip(-5., 5.), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    predictions = []
    for mode in MODES:
        torch.manual_seed(950)
        generator = np.random.default_rng(229)
        net = RidgeCurves(x.shape[1], mode)
        optimizer = torch.optim.Adam(net.parameters(), lr=.008)
        for epoch in range(26):
            for chunk in np.array_split(generator.permutation(train), int(np.ceil(len(train)/256))):
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(net(z[chunk]), target[chunk])
                curvature = net.height[:, 2:] + net.height[:, :-2] - 2 * net.height[:, 1:-1]
                (loss + .025 * curvature.square().mean() + .001 * net.linear.weight.square().sum()).backward()
                optimizer.step()
        with torch.inference_mode():
            predictions.append(torch.sigmoid(net(z[query])).numpy())
    return np.column_stack(predictions)


def run():
    raw, y, group, fold = data()
    names, half, bank = banks(raw, y, fold)
    reference = latest_reference(y, group, fold, names, half, bank)
    _, old_y, old_group, rows, old_fold, _, champion, _, _, _, _, old, _ = aligned()
    score = bank[:, names.index('mitra96')]
    with np.load(ROOT/'artifacts/nov24_remaining_cases/after_corroboration.npz') as cache:
        for key, expected in [('passenger', raw.PassengerId), ('fold', fold), ('y', y)]:
            np.testing.assert_array_equal(cache[key], expected)
        hard = cache['remaining']
    assert hard.sum() == 147 and np.all(reference[hard] != y[hard])
    x = design(raw)
    full_p = protocol(x, y, group, np.arange(len(y)), fold, 'full', predict, MODES, TAG)
    hist_p = protocol(x, y, group, rows, old_fold, 'historical', predict, MODES, TAG)
    new = ~np.isin(group, old_group)
    report = {'training': dict(directions=12, knots=9, epochs=26, seed=950),
              'people': len(y), 'baseline_errors': int((reference != y).sum()),
              'common_errors': int(hard.sum()), 'models': {}}
    common_by_model = {}
    for j, mode in enumerate(MODES):
        common_by_model[mode] = (full_p[:, j] >= .5) == y
        for weight in (1., .25):
            def decisions(anchor, previous, alternative):
                blended = (1-weight)*anchor + weight*alternative >= .5
                return blended if weight==1. else np.where((anchor>=.5)==previous, blended, previous)
            held = decisions(score, reference, full_p[:,j])
            hist = decisions(old[:,0], champion, hist_p[:,j])
            report['models'][f'{mode}_{weight}'] = dict(
                all=stats(held, reference, y, group, fold),
                historical=stats(hist, champion, old_y, old_group, old_fold),
                new_orders=stats(held[new], reference[new], y[new], group[new], fold[new]),
                common_corrected=int((hard & (held==y)).sum()),
                repeated_common_fixes=int((hard[rows] & (champion!=old_y) & (hist==old_y) & (held[rows]==old_y)).sum()))
    report['common_new_beyond_fixed_or_linear'] = int((hard & common_by_model['learned_curves']
        & ~common_by_model['fixed_curves'] & ~common_by_model['linear']).sum())
    learned = common_by_model['learned_curves']
    switched = (reference != y) & learned
    other = common_by_model['fixed_curves'] | common_by_model['linear']
    report['learned_exclusive_among_baseline_errors'] = int((switched & ~other).sum())
    report['learned_fixed_in_both_protocols'] = int((switched[rows] & (champion != old_y)
        & ((hist_p[:, MODES.index('learned_curves')] >= .5) == old_y)).sum())
    report['fixed_any_of_three_among_baseline_errors'] = int(((reference != y) & (learned | other)).sum())
    out = ROOT/'reports/nov23_matched_learning_curve'/f'{TAG}.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    for key, item in report['models'].items():
        print(key, item['all']['corrected'], item['all']['harmed'], item['common_corrected'],
              item['historical']['net'], item['new_orders']['net'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run()
