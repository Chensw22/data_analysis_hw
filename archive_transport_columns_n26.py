"""Read passenger transport probabilities from the named, non-matrix archive formats."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax

ROOT = Path(__file__).resolve().parent
DIRECT = {
    'nov26_joint_party_tabicl': ('direct', 'shuffled', 'joint'),
    'nov26_neighbor_booking_composition': ('plain', 'near', 'shifted'),
    'nov26_domain_aware_consensus': ('seq',),
    'nov26_confidence_ordered_booking': ('raw', 'independent', 'conditional', 'shuffled'),
}


@lru_cache(maxsize=1)
def solo_mask():
    ids = pd.read_csv(ROOT/'data/raw/train.csv', usecols=['PassengerId']).PassengerId
    return ~ids.str.split('_').str[0].duplicated(keep=False).to_numpy()


@lru_cache(maxsize=1)
def booking_priors():
    """Use the identical two sources as family_identity_transfer_n26.panels_and_reference."""
    priors = {}
    art = ROOT/'artifacts'
    with np.load(art/'nov25_historical_nine/historical.npz') as z:
        rows, fold, native = z['rows'], z['fold'], z['scores'][:, 4]
        for k in np.unique(fold):
            priors[tuple(rows[fold == k])] = native[fold == k]
    for k in range(8):
        with np.load(art/f'nov25_grouped_peer_teacher/release{k}.npz') as z:
            priors[tuple(z['q'])] = z['native']
    return priors


def joint_booking_probabilities(z):
    """Reconstruct anchored_booking_distribution_n26's half and full joint updates."""
    q, members, ptr, state_ptr = (z[key] for key in ('q', 'members', 'ptr', 'state_ptr'))
    native = booking_priors()[tuple(q)]
    indices = {row: i for i, row in enumerate(q)}
    masses = {mode: z[mode] for mode in ('independent', 'conditional', 'shuffled')}
    out = {}
    for j, (a, b) in enumerate(zip(ptr[:-1], ptr[1:], strict=True)):
        order = members[a:b]
        bits = (np.arange(1 << len(order))[:, None] >> np.arange(len(order))) & 1
        start, stop = state_ptr[j:j+2]
        positions = np.array([indices.get(row, -1) for row in order])
        present = positions >= 0
        dest = positions[present]
        def independent(margins):
            margins = margins.clip(1e-8, 1-1e-8)
            return bits @ np.log(margins) + (1-bits) @ np.log1p(-margins)
        before = masses['independent'][start:stop] @ bits
        neutral = independent(before)
        prior = before.copy()
        prior[present] = native[dest]
        offset = independent(prior)
        for mode in ('conditional', 'shuffled'):
            mass = masses[mode][start:stop]
            marginal = independent(mass @ bits)
            joint = np.log(mass.clip(1e-15))
            for kind, ratio in [('dependence', joint-marginal), ('marginal', marginal-neutral), ('joint', joint-neutral)]:
                for strength in (.5, 1.):
                    name = f'{mode}_{kind}_{strength}'
                    if name not in out:
                        out[name] = np.full(len(q), .5)
                    out[name][dest] = (softmax(offset + strength*ratio) @ bits)[present]
    return out


def read_probability_columns(folder, z):
    """Return only scores with explicit probability semantics in the generating script."""
    tag = folder.name
    if tag in DIRECT:
        return {key: z[key] for key in DIRECT[tag]}
    if tag == 'nov26_anchored_booking_distribution':
        return joint_booking_probabilities(z)
    if tag == 'nov26_conditional_bill_deviations':
        return {key: expit(z[key]) for key in ('prior', 'raw', 'deviation', 'local_sham', 'same_fit_other_bill')}
    if tag in ('nov26_constrained_track_odds', 'nov26_latent_receipt_channel'):
        keys = ('real', 'sham') if tag == 'nov26_constrained_track_odds' else ('channel', 'template', 'placebo')
        prior = z['prior']
        return {'prior': expit(prior), **{key: expit(prior + z[key]) for key in keys}}
    if tag == 'nov26_cross_deck_budget_multiples':
        names = ('direct', 'cross_deck', 'multiples', 'noninteger')
        effect, prior = z['effect'], z['prior']
        return {'prior': prior, **{key: expit(logit(prior) + effect[:, i]) for i, key in enumerate(names)}}
    if tag == 'nov26_conditional_age_density':
        prior = z['prior']
        out = {'prior': prior, 'direct': z['direct']}
        for key in ('true_year', 'shuffled_year'):
            mass = z[key].clip(1e-9, 1)
            score = expit(logit(prior) + np.log(mass[:, 1]/mass[:, 0]))
            out[key] = np.where(z['age'] >= 0, score, .5)
        return out
    if tag == 'nov26_solo_selection_likelihood':
        blind = z['blind']
        single = solo_mask()[z['q']]
        out = {'blind': blind, 'direct': z['direct']}
        for key in ('true', 'shuffled'):
            a, b = [z[f'{key}_{i}'].clip(.002, .998) for i in (0, 1)]
            ratio = np.where(single, np.log(b/a), np.log((1-b)/(1-a)))
            out[key] = expit(logit(blind) + ratio)
        return out
    if tag == 'nov26_solo_peer_subclasses':
        # Axis 1 is the focal outcome; axis 2 is the latent peer outcome.
        return {key: z[key][:, 1, :].sum(1) for key in ('plain', 'real', 'shuffled')}
    if tag == 'nov26_manifest_role_joint':
        out = {'direct': z['direct']}
        role = z['role']
        for key in ('real', 'sham'):
            joint = z[key]
            mass = joint[np.arange(len(role)), role]
            out[key+'_conditional'] = mass[:, 1] / mass.sum(1)
            out[key+'_marginal'] = joint[:, :, 1].sum(1)
        return out
    if tag == 'nov26_joint_manifest_forest':
        keys = ('service_pair', 'route', 'sleep_side', 'origin_berth')
        values = np.stack([z[key] for key in keys], axis=1)
        out = {'direct': z['direct'], 'masked_mean': values[:, :, 0].mean(1), 'restored_mean': values[:, :, 1].mean(1)}
        for i, key in enumerate(keys):
            out[key+'_masked'] = values[:, i, 0]
            out[key+'_restored'] = values[:, i, 1]
        return out
    return {}
