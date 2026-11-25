"""Apply the unchanged directional gain formula to seven completed November 25 releases."""
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import certified_bill_margins_n25 as bill
from certified_tree_release_n25 import structural_inputs
from edit_population_transfer_n25 import NAMES as FROZEN, load_predictions
from export_consensus_review_n26 import dig
from export_selected_tree_pair_n25 import decisions
from family_identity_transfer_n26 import panels_and_reference
from forecast_frozen_queue_n26 import gains
from full_roster_booking_validation_n23 import ROOT
from full_roster_repartition_gate_n24 import panel_data
from paired_mitra_replicas_n25 import routes
from production_anchor_cpu_replay_n25 import certified, get_challenger
from six_way_against_scored_nine_n26 import test_decisions

ART=ROOT/'artifacts'
REPORT=ROOT/'reports/nov23_matched_learning_curve'
BEST='proposal_n24_selected_dissent'
PAIR='proposal_n25_selected_tree_pair'
CERT='proposal_n25_certified_tree'
BILL='proposal_n25_certified_bill_margins'
REP='proposal_n25_replica_supported_dissent'
NAMES=[PAIR,CERT,BILL,REP,*FROZEN]


def official(name, digest):
    path,=(ROOT/'reports'/name/'online_attempts').glob('20261125T*.json')
    receipt=json.loads(path.read_text())
    e=receipt['evaluation']
    assert e['Status']=='COMPLETE' and e['Submission Ref']==receipt['submission_ref'] and receipt['file_sha256']==digest
    assert receipt['run']==name and receipt['rows']==4277 and e['Submission Date'].startswith('2026-11-25')
    return dict(ref=e['Submission Ref'],score=float(e['Public Score']),sha256=digest)


def directions(edit,base,y,groups,scope,target,test_rows):
    counts=[]; signed=[]
    _,booking=np.unique(groups[scope],return_inverse=True)
    for new in (False,True):
        local=edit[scope] & (base[scope]!=new)
        delta=np.where(base[scope]!=y[scope],1,-1)
        counts.append(np.bincount(booking,weights=local))
        signed.append(np.bincount(booking,weights=local*delta))
    counts=np.column_stack(counts)
    signed=np.column_stack(signed)
    den=counts.sum(0); num=signed.sum(0)
    point=float(gains(num,den,target,test_rows))
    rng=np.random.default_rng(24719)
    samples=[]
    for _ in range(100):
        i=rng.integers(len(counts),size=(20,len(counts)))
        d=counts[i].sum(1)
        a=signed[i].sum(1)
        keep=~((d==0)&(target>0)).any(1)
        samples.extend(gains(a[keep],d[keep],target,test_rows))
    return dict(rows=int(scope.sum()),down_up_local_edits=den.astype(int).tolist(),
                down_up_fixed=((num+den)/2).astype(int).tolist(),down_up_harmed=((den-num)/2).astype(int).tolist(),
                forecast_delta_pp=point,order_bootstrap_gain_pp=np.quantile(samples,[.025,.975]).tolist(),
                valid_draws=len(samples))


def run():
    frozen_path=REPORT/'frozen_queue_forecast.json'
    frozen_sha=dig(frozen_path)
    raw,y,g,f,names,h,full,_,_,cols,rot=panel_data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=lambda c:c!='Name')
    roster=pd.concat([raw,test],ignore_index=True)
    files={n:test_decisions(n,test.PassengerId) for n in [BEST,*NAMES]}
    base,digest=files[BEST]
    queries={}
    for n in NAMES:
        hard,_=files[n]
        queries[n]=np.array([((hard!=base)&(hard==new)).sum() for new in (False,True)],int)
        manifest=json.loads((ROOT/'reports'/n/'manifest.json').read_text())
        assert manifest['edits_vs_best']==queries[n].sum()
        if 'edits_to_true' in manifest:
            assert manifest['edits_to_true']==queries[n][1]
    exact=panels_and_reference()[-1]
    with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['original_prior'],exact[0])
        np.testing.assert_array_equal(z['fold'],f)
        prior=z['prior']
    valid=prior>=0
    nine=np.where(prior==1,exact[2],exact[1]).astype(bool)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,old,old_prior,oh,of=[z[k] for k in ('rows','nine','v131','scores','fold')]
        np.testing.assert_array_equal(z['y'],y[rows])
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[rows])
    j=names.index('mitra96')
    replicas=np.column_stack([h[:,j],rot[:,cols.index(j)]])
    num=full[:,j]; ot=oh[:,4]
    rival=get_challenger(y,g)
    fields=structural_inputs(roster)
    allrows=np.arange(len(y))
    cert=certified(num,*[a[allrows] for a in fields])
    oldcert=certified(ot,*[a[rows] for a in fields])
    assert (valid.sum(),cert.sum(),oldcert.sum())==(8532,4111,3300)
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['certified_v131_equals_v96'],cert)
        np.testing.assert_array_equal(nine[cert],z['certified_nine'][cert])
    assert np.all(valid[cert])
    # Reuse the identical bill models, and fail instead of fitting missing cache entries.
    xs,xt=bill.features(raw,test)
    tg=test.PassengerId.str.split('_').str[0].to_numpy()
    bill_score=[]; models=[]
    for seed in bill.SEEDS:
        assert all((bill.CACHE/f'{seed}_{k}.joblib').exists() for k in range(8))
        prob,parts=bill.fit_train(xs,y,g,seed)
        bill_score.append(np.median(prob,axis=1))
        models.append(parts)
    bprob=np.mean(bill_score,axis=0)
    btest=np.median(bill.test_on_models(models,xt,tg),axis=2).mean(0)
    nt=bill.source('proposal_online_first_counts15',test)[0]
    tcert=certified(nt,*[a[len(y):] for a in fields])
    np.testing.assert_array_equal(bill.edits(nt,btest,base,tcert)[0],files[BILL][0]!=base)
    prior_test=test_decisions('proposal_v131_party_consensus',test.PassengerId)[0]
    frozen_hist,_=load_predictions('historical',roster,rows,of,old,old_prior)
    frozen_test,_=load_predictions('test',roster,np.arange(len(y),len(roster)),np.zeros(len(test),int),base,prior_test)
    for n,a in zip(FROZEN,frozen_test.T,strict=True):
        np.testing.assert_array_equal(a,files[n][0]!=base)
    np.testing.assert_array_equal(tcert&(files[PAIR][0]!=base),files[CERT][0]!=base)
    rtree=json.loads((REPORT/'certified_tree_release.json').read_text())
    rbill=json.loads((REPORT/'certified_bill_margins.json').read_text())['two_fit_mean']
    rhist=json.loads((REPORT/'edit_population_transfer.json').read_text())['results']
    forecasts={k:{} for k in NAMES}
    for tag,idx,bp,pr,nat,c,mask in [
        ('eight_known',allrows,nine,prior.astype(bool),num,cert,valid),
        ('historical',rows,old,old_prior,ot,oldcert,np.ones(len(rows),bool))]:
        pair=(decisions(rival[idx],nat,bp,pr,bp.astype(float))>=.5)!=bp
        confirm=routes(bp,nat,replicas[idx],rival[idx],pr)['tree_without_any_replica']
        edits={PAIR:pair,CERT:c&pair,REP:c&confirm,BILL:bill.edits(nat,bprob[idx],bp,c)[0]}
        if tag=='historical':
            edits.update(dict(zip(FROZEN,frozen_hist.T,strict=True)))
        for n,a in edits.items():
            stats=directions(a,bp,y[idx],g[idx],mask,queries[n],len(test))
            if n in (CERT,REP):
                expected=rtree['full' if tag=='eight_known' else tag]['all_offers' if n==CERT else 'replica_filter']
                assert (sum(stats['down_up_fixed']),sum(stats['down_up_harmed']))==(expected['corrected'],expected['harmed'])
            if n==BILL:
                expected=rbill['eight' if tag=='eight_known' else tag]
                assert (sum(stats['down_up_fixed']),sum(stats['down_up_harmed']))==(expected['fixed'],expected['harmed'])
            if n in FROZEN:
                np.testing.assert_allclose(stats['forecast_delta_pp'],rhist[n]['direction_matched']['expected_accuracy_delta_pp'],rtol=0,atol=1e-12)
            forecasts[n][tag]=stats
    # The older selected-tree audit used a proxy incumbent across all 8693. Keep a
    # separate replication of its directional counts rather than pooling it with exact replays.
    tree_audit=json.loads((REPORT/'selected_tree_pair_transfer.json').read_text())['cohorts']
    proxy=[tree_audit[n] for n in ('originally_true','complement_originally_true')]
    den=np.array([a['corrected']+a['harmed'] for a in proxy])
    num=np.array([a['corrected']-a['harmed'] for a in proxy])
    np.testing.assert_array_equal([a['test_stages'][-1] for a in proxy],queries[PAIR])
    forecasts[PAIR]['earlier_eight_proxy_delta_pp']=float(gains(num,den,queries[PAIR],len(test)))
    # Join official scores only after the forecasts and their independent replay checks are fixed.
    receipts={n:official(n,files[n][1]) for n in [BEST,*NAMES]}
    best=receipts[BEST]['score']
    known=json.loads(frozen_path.read_text())
    assert known['starting_public_score']==best and known['reference']==BEST and known['official_ref']==receipts[BEST]['ref']
    summary={}
    for n in NAMES:
        actual=100*(receipts[n]['score']-best)
        for tag in ('eight_known','historical'):
            if tag not in forecasts[n]: continue
            v=forecasts[n][tag]
            p=v['forecast_delta_pp']
            lo,hi=v['order_bootstrap_gain_pp']
            v.update(predicted_public=best+p/100,error_pp=p-actual,direction_correct=bool(np.sign(actual)==np.sign(p)),
                     actual_within_local_bootstrap=bool(lo<=actual<=hi))
        forecasts[n].update(receipt=receipts[n],down_up_test_edits=queries[n].tolist(),actual_delta_pp=actual)
    for tag in ('eight_known','historical'):
        vv=[v[tag] for v in forecasts.values() if tag in v]
        summary[tag]=dict(versions=len(vv),direction_correct=sum(v['direction_correct'] for v in vv),
                          mean_absolute_error_pp=float(np.mean([abs(v['error_pp']) for v in vv])),
                          mean_signed_overestimate_pp=float(np.mean([v['error_pp'] for v in vv])),
                          inside_local_bootstrap=sum(v['actual_within_local_bootstrap'] for v in vv),
                          unchanged_score_control_mae_pp=float(np.mean([abs(v['actual_delta_pp']) for v in forecasts.values() if tag in v])))
    report=dict(reference=receipts[BEST],scored_versions=forecasts,summary=summary,
          scope='Retrospective replay of the November 26 formula on the seven N25 releases relative to the earlier scored nine-vote incumbent. No test outcomes used for fitting; only official public aggregate outcomes are joined.',
          protocols='Historical nine-vote parent: 6,954 people, three original passenger folds (2,263 with a peer in another fold). Eightfold parent: 8,532 of 8,693 known exact decisions on whole-order folds. Tree challengers use two five-fold order replays, replicas use independent half-roster assignments, bill challengers use two eight-fold order partitions. These different fits all exclude the focal passenger. There is no matching eightfold challenger replay for the last three frozen binary exports.',
          limitations='Related submissions sharing an incumbent and a single public scoreboard are not independent draws. The empirical sign failures cannot estimate the win probability of unrelated candidates. The unchanged-score control uses only the public incumbent.',
          bootstrap='Frozen booking bootstrap, 2000 draws with seed 24719. Per-version intervals and valid directional draws are computed separately because not every family exists on both populations. No private scores or individual test labels are used.')
    assert dig(frozen_path)==frozen_sha
    (REPORT/'backtest_frozen_queue_forecast.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary)
    for n,v in forecasts.items():
        print(n,[(tag,round(v[tag]['forecast_delta_pp'],3)) for tag in ('eight_known','historical') if tag in v],round(v['actual_delta_pp'],3))


if __name__=='__main__':
    with threadpool_limits(2): run()
