"""Estimate booking-held-out calibration shifts from nested teacher errors in overlapping passenger groups."""
import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST, stats
from full_roster_booking_validation_n23 import data
from full_roster_six_failure_archive_n24 import prediction_sets
from grouped_peer_release_replay_n25 import edit_effect_bounds
from grouped_peer_teacher_n25 import OUT

BILLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
METHODS = ('global', 'single', 'cohorts', 'cohorts_half', 'shuffled_cohorts')


def groups(frame):
    fees = frame[BILLS].fillna(0)
    cabin = frame.Cabin.str.split('/', expand=True)
    booking = frame.PassengerId.str.split('_').str[0]
    total = fees.sum(axis=1)
    dominant = fees.idxmax(axis=1).where(total > 0, 'none')
    fields = {k: frame[k] for k in ('HomePlanet', 'Destination', 'VIP', 'CryoSleep')}
    fields.update(deck=cabin[0], side=cabin[2], age=pd.cut(frame.Age, [-1,12,17,25,39,59,200]),
                  booking=booking.map(booking.value_counts()).clip(upper=3),
                  services=fees.gt(0).sum(axis=1).clip(upper=3), dominant=dominant,
                  total=pd.cut(total, [-1,0,300,800,1200,2500,5000,np.inf]),
                  missing=frame.drop(columns='PassengerId').isna().sum(axis=1).clip(upper=2))
    for name in BILLS:
        fields[name] = pd.cut(frame[name], [-1,0,300,800,1500,np.inf])
    return {k: v.astype('string').fillna('?').to_numpy(str) for k,v in fields.items()}


def matrix(fields, training):
    names, columns = ['all'], [np.arange(len(next(iter(fields.values()))))]
    singles = 1
    for order in (1,2):
        for keys in combinations(fields,order):
            value = fields[keys[0]]
            if order == 2:
                value = np.char.add(np.char.add(value, '/'), fields[keys[1]])
            levels, idx = np.unique(value,return_inverse=True)
            for j in np.flatnonzero(np.bincount(idx[training],minlength=len(levels)) >= 80):
                columns.append(np.flatnonzero(idx == j))
                names.append('+'.join(keys)+'='+levels[j])
        if order == 1:
            singles = len(columns)
    indices = np.concatenate(columns)
    offset = np.r_[0,np.cumsum([len(c) for c in columns])]
    design = csc_matrix((np.ones(len(indices)),indices,offset),shape=(len(value),len(names)))
    return design, singles, names


def shifts(matrix, p, y, t, q):
    train, query = matrix[t].tocsc(), matrix[q].tocsc()
    scores = logit(np.clip(p,.002,.998))
    correction = np.zeros(len(q))
    chosen = []
    # Shrink corrections in small and/or low-variance groups. Select only with outer training labels.
    for _ in range(24):
        pred = expit(scores)
        residual = train.T @ (y-pred)
        curvature = train.T @ (pred*(1-pred)) + 12
        j = int(np.argmax(np.abs(residual)/np.sqrt(curvature)))
        step = float(np.clip(.5*residual[j]/curvature[j],-.8,.8))
        i, end = train.indptr[j:j+2]
        scores[train.indices[i:end]] += step
        i, end = query.indptr[j:j+2]
        correction[query.indices[i:end]] += step
        chosen.append((j,step))
    return correction, chosen


def run(folds):
    raw, y, booking, fold = data()
    fields = groups(raw)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf = z['rows'], z['fold']
    _, panels = prediction_sets(raw,y,booking,fold,rows,hf)
    votes = (np.column_stack(panels[0]) >= .5).sum(axis=1)
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['fold'],fold)
        native = z['v96'].astype(float)
    data_by_fold = []
    paths = {}
    for k in folds:
        with np.load(OUT/f'outer{k}.npz') as z:
            t,q,teacher = z['t'],z['q'],z['p']
            np.testing.assert_array_equal(z['labels'],y[t])
            np.testing.assert_array_equal(t,np.flatnonzero(fold!=k))
            np.testing.assert_array_equal(q,np.flatnonzero(fold==k))
            assert not np.isin(booking[q],booking[t]).any()
        with np.load(OUT/f'release{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            parent,nine,low,high = (z[j] for j in ('v131','nine','nine_on_low_branch','nine_on_high_branch'))
        m,width,names = matrix(fields,t)
        permuted=y[t].copy()
        bins = np.digitize(teacher,np.arange(.05,1,.05))
        rng=np.random.default_rng(692+k)
        for b in np.unique(bins):
            mask=bins==b
            permuted[mask]=rng.permutation(permuted[mask])
        paths[k] = {}
        offsets = {}
        for key,size,labels in [('global',1,y[t]),('single',width,y[t]),('cohorts',m.shape[1],y[t]),
                                ('shuffled_cohorts',m.shape[1],permuted)]:
            offsets[key], steps = shifts(m[:,:size], teacher,labels,t,q)
            paths[k][key] = [(names[j],round(step,4)) for j,step in steps]
        offsets['cohorts_half']=offsets['cohorts']/2
        record = {}
        for key in METHODS:
            pred = logit(native[q].clip(.002,.998))+offsets[key] >= 0
            a,b = edit_effect_bounds(pred,native[q]>=.5,parent,low,high,y[q])
            record[key] = (pred,a,b)
        data_by_fold.append((q,parent,nine,record))
        print('evaluated fold',k,flush=True)
    q,parent,nine = (np.concatenate([v[i] for v in data_by_fold]) for i in range(3))
    known = parent>=0
    unanimous = (votes[q]==0)|(votes[q]==68)
    majority = votes[q]>34
    report = {}
    for key in METHODS:
        pred,a,b = (np.concatenate([v[3][key][i] for v in data_by_fold]) for i in range(3))
        agree = majority==nine
        direct = (pred==y[q]).astype(int)-(majority==y[q])
        assert (a[known]==b[known]).all()
        report[key] = dict(net_bounds=[int(a.sum()),int(b.sum())],
                           known=stats(a,known,fold[q],booking[q]),
                           remaining_five=stats(a,known&(fold[q]>=3),fold[q],booking[q]) if any(f>=3 for f in folds) else None,
                           unseen_orders=stats(a,known&~np.isin(booking[q],booking[rows]),fold[q],booking[q]),
                           unanimous_nine=stats(a,known&agree&unanimous,fold[q],booking[q]),
                           direct_unanimous=stats(direct,unanimous,fold[q],booking[q]))
    name = 'booking_multicalibration' if len(folds)==8 else 'booking_multicalibration_'+'_'.join(map(str,folds))
    path = DEST.with_name(name+'.json') if len(folds)==8 else OUT/(name+'.json')
    path.write_text(json.dumps(dict(folds=folds, rows=len(q), known=int(known.sum()), results=report,
                                    training_steps=paths),indent=2)+'\n')
    for key,r in report.items():
        print(key,r['net_bounds'],r['known']['fixed'],r['known']['harmed'],r['direct_unanimous']['fixed'],
              r['direct_unanimous']['harmed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folds',nargs='+',type=int,choices=range(8),default=list(range(8)))
    with threadpool_limits(2):
        run(parser.parse_args().folds)
