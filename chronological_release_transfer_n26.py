"""Audit whole-order validation and a prior-day error correction using aggregate evaluation receipts."""
import json

import numpy as np
from scipy.stats import spearmanr

from full_roster_booking_validation_n23 import ROOT


FOLDER = ROOT/'reports/nov23_matched_learning_curve'


def read(name):
    return json.loads((FOLDER/name).read_text())


def describe(predictions, actual):
    keys = [k for k in actual if predictions.get(k) is not None]
    a = np.array([actual[k] for k in keys])
    p = np.array([predictions[k] for k in keys])
    return dict(versions=len(keys),mean_error_pp=float(np.mean(p-a)),mean_absolute_error_pp=float(np.mean(abs(p-a))),
                predicted_gains=int((p>0).sum()),sign_correct=int((np.sign(a)==np.sign(p)).sum()),
                spearman=float(spearmanr(a,p).statistic) if len(np.unique(p))>1 and len(keys)>1 else None)


def run():
    old = read('backtest_frozen_queue_forecast.json')
    today = read('nov26_official_queue_results.json')
    forecast = read('frozen_queue_forecast.json')
    mix = read('frozen_release_order_mix.json')
    n = forecast['test_rows']
    assert n==mix['test_passengers'] and old['reference']['score']==today['incumbent_public']
    actual = {k:r['public_gain_pp'] for k,r in today['scores'].items() if k!='reserve'}
    edits = {k:forecast['versions'][k]['changed_on_test'] for k in actual}
    raw = {k:forecast['versions'][k]['conditional_gain_pp'] for k in actual}
    methods = dict(fixed_incumbent={k:0. for k in actual},frozen_local=raw)
    prior = {}
    for scope in ['historical','eight_known']:
        batch=[r for r in old['scored_versions'].values() if scope in r]
        volume=sum(sum(r['down_up_test_edits']) for r in batch)
        public_slope=sum(r['actual_delta_pp'] for r in batch)/volume
        error_slope=sum(r[scope]['forecast_delta_pp']-r['actual_delta_pp'] for r in batch)/volume
        prior[scope]=dict(versions=len(batch),edits_across_files=volume,public_gain_pp_per_edit=public_slope,
                          local_overestimate_pp_per_edit=error_slope)
        methods[scope+'_prior_edit_loss']={k:v*public_slope for k,v in edits.items()}
        methods[scope+'_subtract_prior_optimism']={k:raw[k]-edits[k]*error_slope for k in actual}
    for scope,c in mix['cohorts'].items():
        methods[scope+'_raw']={k:c['candidates'][k]['raw_gain_pp'] for k in actual}
        methods[scope+'_size_and_direction']={k:c['candidates'][k]['point_gain_pp'] for k in actual}
    measured={k:describe(v,actual) for k,v in methods.items() if any(x is not None for x in v.values())}
    for k in methods.keys()-measured.keys():
        measured[k]=dict(versions=0,reason='All candidates have at least one edited test stratum absent from this source.')
    # The chain below uses only single known parents or restriction of the same previous branch.
    # The three-way union has two parents; its filtered proxy cannot replace an unsubmitted parent.
    pairs=[('awake','filtered'),('all_three','review'),('review','fare'),('fare','six'),
           ('six','tabnet'),('tabnet','reserve')]
    chain={}
    base=mix['cohorts']['all_eight_folds']['candidates']
    for a,b in pairs:
        la=base[b]['validation_net']-base[a]['validation_net']
        pu=100*(today['scores'][b]['public']-today['scores'][a]['public'])
        chain[f'{a}_to_{b}']=dict(local_net=la,public_gain_pp=pu,same_direction=bool(np.sign(la)==np.sign(pu)))
    output=dict(prior=prior,performance=measured,per_candidate={k:{name:v[k] for name,v in methods.items()} for k in actual},
                direct_stage_comparisons=chain,public_gain_pp=actual,
                scope='Retrospective diagnostics. Slopes are estimated solely on Nov 25 aggregate receipts. Related submissions share an incumbent and changed passengers. No individual test labels are used. A missing test stratum is never treated as a zero-effect estimate.')
    (FOLDER/'chronological_release_transfer.json').write_text(json.dumps(output,indent=2)+'\n')
    print({k:(v['versions'],round(v.get('mean_absolute_error_pp',0),3),v.get('predicted_gains')) for k,v in measured.items()})
    print(chain)


if __name__=='__main__':
    run()
