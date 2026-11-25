"""Learn from the exact deployed native teacher's errors on other held-out bookings of the same fold."""
import argparse
import json
import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold

from full_roster_booking_validation_n23 import ROOT,data
from matched_error_ranker_n25 import SOURCE, MODES, covariates, train_predict

OUT=ROOT/'artifacts/nov25_same_teacher_error_pairs'
REPORT=ROOT/'reports/nov23_matched_learning_curve/same_teacher_error_pairs.json'


def generate():
    import torch
    torch.set_num_threads(2)
    raw,y,g,f=data()
    with np.load(ROOT/'artifacts/nov25_historical_nine/historical.npz') as z:
        fresh=~np.isin(g,g[z['rows']])
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        q,t=np.flatnonzero(f==k),np.flatnonzero(f!=k)
        with np.load(SOURCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            np.testing.assert_array_equal(z['y'],y[q])
            native=z['native']
        assert not np.isin(g[t],g[q]).any()
        frame=covariates(raw,g,t,q,k)[q]
        splits={'all':list(GroupKFold(4,shuffle=True,random_state=3207).split(q,groups=g[q])),
                'new':[(np.flatnonzero(~fresh[q]),np.flatnonzero(fresh[q]))]}
        for protocol,parts in splits.items():
            for j,(a,b) in enumerate(parts):
                assert not np.isin(g[q[a]],g[q[b]]).any()
                path=OUT/f'{k}_{protocol}_{j}.npz'
                if path.exists():
                    continue
                matrix=frame[np.r_[a,b]]
                values={m:train_predict(matrix,native[a],native[b],y[q[a]],g[q[a]],m,3641+k*11+j) for m in MODES}
                assert all(np.isfinite(v).all() for v in values.values())
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    np.savez_compressed(stream,t=q[a],q=q[b],calibration_labels=y[q[a]],native=native[b],**values)
                tmp.replace(path)
        print('same native teacher',k,flush=True)


def evaluate():
    from adjudicate_amount_dissent_n25 import counts
    from disjoint_tree_fields_n25 import ART
    from full_roster_six_failure_archive_n24 import prediction_sets
    from grouped_peer_release_replay_n25 import edit_effect_bounds
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    votes,old=[(np.column_stack(panel)>=.5).sum(1) for panel in panels]
    common=(votes==0)|(votes==68)
    repeated=np.zeros(len(y),bool)
    repeated[rows]=common[rows]&((old==0)|(old==68))&((old>34)==(votes[rows]>34))
    new=~np.isin(g,g[rows])
    parts={}
    for k in range(8):
        fold_rows=np.flatnonzero(f==k)
        with np.load(SOURCE/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],fold_rows)
            p,prior,lo,hi=[z[n] for n in ('native','v131','nine_on_low_branch','nine_on_high_branch')]
        for protocol,blocks in [('all',4),('new',1)]:
            for j in range(blocks):
                with np.load(OUT/f'{k}_{protocol}_{j}.npz') as z:
                    t,q=z['t'],z['q']
                    ix=np.searchsorted(fold_rows,q)
                    np.testing.assert_array_equal(fold_rows[ix],q)
                    np.testing.assert_array_equal(z['native'],p[ix])
                    np.testing.assert_array_equal(z['calibration_labels'],y[t])
                    assert (f[t]==k).all() and not np.isin(g[t],g[q]).any()
                    if protocol=='new':
                        assert not new[t].any() and new[q].all()
                    native=p[ix]>=.5
                    for m in MODES:
                        for scale in (.5,1.):
                            change=scale*z[m]>np.abs(logit(p[ix].clip(.002,.998)))
                            a,b=edit_effect_bounds(native^change,native,prior[ix],lo[ix],hi[ix],y[q])
                            known=prior[ix]>=0
                            np.testing.assert_array_equal(a[known],b[known])
                            parts.setdefault((protocol,m,scale),[]).append((q,known,a,b))
    report={}
    for (protocol,m,scale),blocks in parts.items():
        q,known,a,b=(np.concatenate([block[i] for block in blocks]) for i in range(4))
        np.testing.assert_array_equal(np.sort(q),np.arange(len(y)) if protocol=='all' else np.flatnonzero(new))
        r=dict(bounds=[int(a.sum()),int(b.sum())])
        for key,mask in [('known',known),('common',known&common[q]),('repeated',known&repeated[q]),
                         ('common_false',known&(votes[q]==0)),('common_true',known&(votes[q]==68))]:
            r[key]=counts(a,mask,f[q])
        report.setdefault(protocol,{})[f'{m}/{scale}']=r
        print(protocol,m,scale,r['bounds'],[(key,r[key]['fixed'],r[key]['harmed']) for key in ('known','common')])
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['generate','evaluate'])
    generate() if p.parse_args().action=='generate' else evaluate()
