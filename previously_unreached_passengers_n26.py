"""Quantify how many errors new model families reach beyond the entire previous opinion archive."""
import argparse
import json

import numpy as np
from threadpoolctl import threadpool_limits

from distinct_challenger_error_reach_n26 import population
from exhaustive_family_quorums_n26 import inputs
from joint_new_architectures_n26 import ART, REPORT
from rtj_checkpoint_setup_n26 import WEIGHTS


def get_votes(raw, y, g, f, with_rtj):
    ids = raw.PassengerId.to_numpy(str)
    models = {}
    for file, keys in [('nov26_sparse_boolean_clauses/eight_fold_oof.npz', {'clauses':'full'}),
                       ('nov26_relaxed_passenger_transport/eight_fold_oof.npz', {'soft_matching':'relaxed'}),
                       ('nov26_nori/aligned.npz', {'Nori384':'context_384', 'Nori1536':'context_1536', 'Nori_full':'full'})]:
        with np.load(ART / file) as z:
            for key, v in [('passenger',ids),('row',np.arange(len(y))),('fold',f),('y',y)]:
                np.testing.assert_array_equal(z[key],v)
            for name, key in keys.items():
                assert np.isfinite(z[key]).all()
                models[name] = z[key] >= .5
    capsules = np.zeros(len(y), bool)
    rtj = {m:np.zeros(len(y), bool) for m in ['edges','no_edges']}
    for k in np.unique(f):
        t, q = np.flatnonzero(f != k), np.flatnonzero(f == k)
        assert not np.isin(g[t], g[q]).any()
        p = []
        for seed in [4537,8912]:
            with np.load(ART / f'nov26_booking_capsules/fold{k}_{seed}.npz') as z:
                for key, v in [('t',t),('q',q),('labels',y[t]),('ids',ids[q])]:
                    np.testing.assert_array_equal(z[key],v)
                p.append(z['dynamic'])
        capsules[q] = np.mean(p,axis=0) >= .5
        if with_rtj:
            for mode, arr in rtj.items():
                with np.load(ART / f'nov26_rtj_probe/oof/{k}_{mode}.npz') as z:
                    np.testing.assert_array_equal(z['ids'], ids[q])
                    np.testing.assert_array_equal(z['y'], y[q])
                    assert np.isfinite(z['logits']).all() and str(z['weights']) == WEIGHTS
                    arr[q] = z['logits'] > 0
    models['capsules'] = capsules
    if with_rtj:
        models.update({f'RTJ_{mode}':p for mode,p in rtj.items()})
    return models


def counts(mask, wrong):
    return dict(rows=int(mask.sum()), wrong=int((mask & wrong).sum()), correct=int((mask & ~wrong).sum()))


def run(with_rtj):
    raw,y,g,f,old,same,blocks,_,_,_,solo,paid = population()
    _,cur,_ = inputs()
    parent = cur['parent']
    np.testing.assert_array_equal(parent[cur['common']],old[cur['common']])
    assert not (cur['common'] & ~same).any()
    models = get_votes(raw,y,g,f,with_rtj)
    names = list(models)
    proposals = np.column_stack(list(models.values())) != parent[:,None]
    any_new = proposals.any(axis=1)
    wrong = parent != y
    silent = cur['common'] & (blocks == 0)
    report = dict(previous_model_outputs=127, new_models=names, scopes={}, mutually_exclusive_manifest_groups={})
    for name,scope in [('all_common',cur['common']),('no_previous_model_challenges',silent)]:
        families = dict(nori=proposals[:,[n.startswith('Nori') for n in names]].any(1),
                        boolean=proposals[:,names.index('clauses')],matching=proposals[:,names.index('soft_matching')],
                        capsules=proposals[:,names.index('capsules')])
        if with_rtj:
            families['RTJ'] = proposals[:,[n.startswith('RTJ') for n in names]].any(1)
        votes = np.column_stack(list(families.values())).sum(1)
        report['scopes'][name] = dict(total=counts(scope,wrong),unchallenged=counts(scope & ~any_new,wrong),
            each={n:counts(scope & proposals[:,j],wrong) for j,n in enumerate(names)},
            family_only={n:counts(scope & m & (votes==1),wrong) for n,m in families.items()},
            by_family_count={str(j):counts(scope & (votes==j),wrong) for j in range(len(families)+1)})
    sleeper = raw.CryoSleep.eq(True).to_numpy() & ~paid
    awake = raw.CryoSleep.eq(False).to_numpy() & paid
    still = silent & ~any_new
    total = 0
    for state, m in [('sleep_no_positive_bill',sleeper),('awake_positive_bill',awake),('other',~sleeper & ~awake)]:
        for sign in [False,True]:
            cell = m & (parent==sign)
            total += int((still & cell).sum())
            report['mutually_exclusive_manifest_groups'][f'{state}_predicted_{sign}'] = dict(
                silent_before=counts(silent & cell,wrong), silent_after=counts(still & cell,wrong),
                solo_after=counts(still & cell & solo,wrong))
    assert total == still.sum()
    path = REPORT.parent / ('previously_unreached_with_rtj.json' if with_rtj else 'previously_unreached_passengers.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print({k:(v['total'],v['unchallenged']) for k,v in report['scopes'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-rtj',action='store_true')
    with threadpool_limits(1):
        run(parser.parse_args().with_rtj)
