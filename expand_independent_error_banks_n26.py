"""Enlarge the labelled reference and calibration sets separately without exposure to query labels."""
import json

import numpy as np
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from conflict_case_control_banks_n26 import predict_model,retrieve
from conflict_pair_ordering_n26 import CACHE,features,fit_predict
from cross_order_conflict_cover_n26 import relations
from exhaustive_family_quorums_n26 import inputs,summary
from full_roster_booking_validation_n23 import ROOT,OUT,data
from full_roster_consensus_ranking_n23 import half_bank

DEST = ROOT/'artifacts/nov26_independent_error_banks'


def bank_inputs(raw,y,g,f):
    half_prob = half_bank(raw,y,f)[:,-1]
    with np.load(OUT/'mitra96.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        full_prob = z['p']
    with np.load(CACHE/'held.npz') as z:
        np.testing.assert_array_equal(z['passenger'],raw.PassengerId.to_numpy(str))
        np.testing.assert_array_equal(z['fold'],f)
        full_rank = z['scores']
    a,b,d = relations(raw,g,False)
    raw_x = features(raw,g)
    half_rank = np.zeros_like(full_rank)
    for parity in (0,1):
        eligible = f%2!=parity
        q = np.flatnonzero(~eligible)
        half_rank[q] = fit_predict(raw_x,y,g,a,b,d,eligible,q)
    values,sides = [],[]
    for p,r in [(half_prob,half_rank),(full_prob,full_rank)]:
        side = p>=.5
        values.append(np.column_stack([logit(p.clip(.003,.997)),r])*np.where(side,-1,1)[:,None])
        sides.append(side)
    keep = f[a]%2==f[b]%2
    a,b,d = a[keep],b[keep],d[keep]
    ix = np.argsort(np.r_[a,b],kind='stable')
    splits = np.cumsum(np.bincount(np.r_[a,b],minlength=len(y)))[:-1]
    return values,sides,np.split(np.r_[b,a][ix],splits),np.split(np.r_[d,d][ix],splits)


def run():
    raw,y,g,f = data()
    _,cur,_ = inputs()
    values,sides,edges,dist = bank_inputs(raw,y,g,f)
    sparse_bank = partition(g,'dense-bank-a:',4)==0
    sparse_cal = partition(g,'dense-cal-a:',4)==0
    report = {}
    DEST.mkdir(exist_ok=True)
    for query_source in (0,1):
        predictions = np.full((len(y),3,2,2,2),np.nan)
        coverage = np.zeros((len(y),3,2),bool)
        source_sizes = []
        for k in np.unique(f):
            q = np.flatnonzero(f==k)
            x = values[0].copy()
            x[q] = values[query_source][q]
            side = sides[0].copy()
            side[q] = sides[query_source][q]
            marked = side!=y
            remaining = np.unique(f[(f%2==k%2)&(f!=k)])
            assert len(remaining)==3
            for rep,cal_fold in enumerate(remaining):
                cal = np.flatnonzero(f==cal_fold)
                large = np.flatnonzero(np.isin(f,remaining[remaining!=cal_fold]))
                assert not np.isin(g[q],g[np.r_[cal,large]]).any() and not np.isin(g[cal],g[large]).any()
                for h,bank in enumerate((large[sparse_bank[large]],large)):
                    scaled = StandardScaler().fit(x[bank]).transform(x).clip(-6,6)
                    ids = np.r_[cal,q]
                    original = np.column_stack([scaled[ids],side[ids]])
                    info,covered = retrieve(ids,bank,edges,dist,scaled,side,marked,g)
                    augmented = np.column_stack([original,info])
                    coverage[q,rep,h] = covered[len(cal):]
                    for j,use in enumerate((np.flatnonzero(sparse_cal[cal]),np.arange(len(cal)))):
                        for m,design in enumerate((original,augmented)):
                            predictions[q,rep,h,j,m] = predict_model(design[use],design[len(cal):],marked[cal[use]])
                        source_sizes.append((h,j,len(bank),len(use)))
        assert np.isfinite(predictions).all()
        valid = cur['known']&(sides[query_source]==cur['parent'])
        tag = ['half_to_half','half_to_full'][query_source]
        report[tag] = {}
        for h in range(2):
            for j in range(2):
                key = ('full_library' if h else 'quarter_library')+'_'+('full_calibration' if j else 'quarter_calibration')
                sizes = np.array([v[2:] for v in source_sizes if v[:2]==(h,j)])
                votes = predictions[:,:,h,j,:].mean(1)>.5
                established = coverage[:,:,h].all(1)
                actions = dict(individual=votes[:,0],retrieved=votes[:,1],both_banks_supported=votes[:,1]&established,
                               additional_when_supported=votes[:,1]&~votes[:,0]&established)
                result = dict(reference_passengers=sizes[:,0].mean(),calibration_passengers=sizes[:,1].mean(),scopes={})
                for n,mask in [('all',valid),('common',valid&cur['common']),('new_orders',valid&cur['new'])]:
                    result['scopes'][n] = {m:summary(act,cur['wrong'],mask,f) for m,act in actions.items()}
                    result['scopes'][n]['covered'] = int((mask&established).sum())
                report[tag][key] = result
                v = result['scopes']['common']
                print(tag,key,v['covered'],v['both_banks_supported']['fixed'],v['both_banks_supported']['harmed'],flush=True)
        np.savez_compressed(DEST/(tag+'.npz'),passenger=raw.PassengerId.to_numpy(str),truth=y,fold=f,
                           scores=predictions,coverage=coverage,side=sides[query_source])
    report['scope'] = ('Reference and calibration passengers were scored using base models trained on the opposite '
       'half of all bookings. The query base uses either that same half or all seven other folds. Library and '
       'calibration sizes vary independently. The query outcomes never enter either source.')
    (ROOT/'reports/nov23_matched_learning_curve/expand_independent_error_banks.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):
        run()
