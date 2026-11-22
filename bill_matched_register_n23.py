"""Outcome-blind bill matching and held-out identification of discordant passengers."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit
from threadpoolctl import threadpool_limits

from boarding_sequence_n23 import ART, ROOT, SEEDS
from conditional_view_density_n23 import BILLS, fit_score
from exaone_validation_data import CATEGORICAL, compact_features
from grouped_five_way_coverage_n23 import verify
from population_rescue_map_n23 import read_panel
from relational_error_corrector import write

LIMIT = np.log(1.25)


def match(bills, group, cabin, rows, seed):
    """Each booking can enter only one pair. Outcomes and registration never enter the distance."""
    rng = np.random.default_rng(seed)
    rows = rng.permutation(rows)
    billed = bills[rows]
    state = np.where(np.isnan(billed), 2, np.where(billed > 0, 1, 0))
    levels, inverse = np.unique(state, axis=0, return_inverse=True)
    # Spend profiles with several observed purchases have the fewest close matches;
    # allocate scarce distinct bookings to them before the large all-zero tied sets.
    blocks = sorted(range(len(levels)), key=lambda k: (-np.sum(levels[k] == 1), (inverse == k).sum(), k))
    used = set()
    pairs = []

    def accept(i, j, distance):
        if group[i] == group[j] or group[i] in used or group[j] in used:
            return False
        if cabin[i] and cabin[i] == cabin[j]:
            return False
        used.update((group[i], group[j]))
        pairs.append((i, j, distance))
        return True

    for k in blocks:
        local = rows[inverse == k]
        paid = levels[k] == 1
        if len(local) < 2:
            continue
        if not paid.any():
            pending = []
            for i in local:
                if group[i] in used:
                    continue
                for j, other in enumerate(pending):
                    if accept(i, other, 0.):
                        pending.pop(j)
                        break
                else:
                    pending.append(i)
            continue
        amounts = np.log1p(bills[local][:, paid])
        neighbors = cKDTree(amounts).query_pairs(LIMIT, p=np.inf, output_type='ndarray')
        dist = np.max(np.abs(amounts[neighbors[:, 0]]-amounts[neighbors[:, 1]]), axis=1)
        for i in np.lexsort((rng.random(len(dist)), dist)):
            a, b = neighbors[i]
            accept(local[a], local[b], dist[i])
    pairs = np.asarray(pairs)
    left, right = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    assert len(np.unique(group[np.r_[left,right]])) == 2*len(left)
    assert not ((cabin[left] != '') & (cabin[left] == cabin[right])).any()
    return left, right, pairs[:, 2]


def score_pairs(y, a, b, score):
    discord = y[a] != y[b]
    delta = (2*y[a[discord]].astype(int)-1)*(score[a[discord]]-score[b[discord]])
    hit = (delta > 1e-12).astype(float) + .5*(np.abs(delta) <= 1e-12)
    return hit, int((np.abs(delta) <= 1e-12).sum())


def summary(y,a,b,dist,scores,subsets):
    report = {}
    for scope, mask in subsets.items():
        aa, bb = a[mask], b[mask]
        disagree = y[aa] != y[bb]
        n = int(disagree.sum())
        measures = {}
        for name, p in scores.items():
            h, ties = score_pairs(y,aa,bb,p)
            measures[name] = dict(numerator=float(h.sum()), ties=ties,
                                  concordance=float(h.mean()) if n else None)
        report[scope] = dict(pairs=len(aa), persons=2*len(aa), discord=n,
                max_gap_p50_p95=[float(v) for v in np.quantile(dist[mask],[.5,.95])] if mask.any() else [],
                scores=measures)
        if n:
            # Within each held-out split, pairs contain distinct bookings; only
            # discordant pairs contribute. Flipping true orientation provides the null.
            u, _ = score_pairs(y,aa,bb,scores['register'])
            e, _ = score_pairs(y,aa,bb,scores['economic'])
            rng = np.random.default_rng(410)
            boot = np.mean((u-e)[rng.integers(0,n,size=(1500,n))], axis=1)
            report[scope]['register_minus_economic'] = [float((u-e).mean()), *np.quantile(boot,[.025,.975]).tolist()]
            report[scope]['register_reversal'] = dict(
                correct_against_economic=int(((u==1)&(e==0)).sum()), wrong_against_economic=int(((u==0)&(e==1)).sum()),
                expected_pair_permutation=.5)
    return report


def run():
    rows = pd.read_csv(ART/'nov23_global_error_inventory.csv')
    row=rows.row.to_numpy()
    y, bh, hh = read_panel()
    np.testing.assert_array_equal(rows.target,y)
    wrong, hard = ~bh.any(0), ~bh.any(0)&~hh.any(axis=(0,1))
    assert [len(y),wrong.sum(),hard.sum()]==[6954,1070,466]
    raw=pd.read_csv(ROOT/'data/raw/train.csv',usecols=lambda c:c not in ['Transported','Name']).iloc[row]
    x=compact_features(raw).reset_index(drop=True)
    group=raw.PassengerId.str.split('_').str[0].to_numpy()
    cabin=raw.Cabin.fillna('').to_numpy()
    bills=x[BILLS].to_numpy(float)
    register = x[CATEGORICAL].copy()
    for c in CATEGORICAL:
        register[c] = pd.Categorical(register[c]).codes
    register['AgeBand'] = np.where(x.Age.isna(), 7, np.digitize(x.Age,[6,13,18,25,40,60]))
    register = register.to_numpy(float)
    economic = np.log1p(bills)
    joint = np.column_stack((register,economic))
    variants = {'register':(register,np.ones(register.shape[1],bool)), 'economic':(economic,None),
                'joint':(joint,np.r_[np.ones(register.shape[1],bool),np.zeros(economic.shape[1],bool)])}
    sleep_col = CATEGORICAL.index('CryoSleep')
    no_sleep = np.delete(register,sleep_col,axis=1)
    variants['register_without_sleep'] = (no_sleep,np.ones(no_sleep.shape[1],bool))
    variants['sleep_only'] = (register[:,[sleep_col]],np.ones(1,bool))
    out=ART/'nov23_bill_matched_register'
    out.mkdir(exist_ok=True)
    final={}
    for seed in SEEDS:
        with np.load(ART/f'nov23_group_kernel/{seed}.npz') as z:
            fold,base=z['fold'],z['base']
            verify(z,row,y,fold)
        predictions={k:np.full(len(y),np.nan) for k in variants}
        left,right,dist=[],[],[]
        for k in np.unique(fold):
            t,q=np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            assert not np.intersect1d(group[t],group[q]).size
            a,b,d=match(bills,group,cabin,q,371+int(k))
            left.extend(a);right.extend(b);dist.extend(d)
            for name,(features,cats) in variants.items():
                predictions[name][q]=expit(fit_score(features[t],y[t],features[q],cats))
        a,b,d=np.asarray(left),np.asarray(right),np.asarray(dist)
        assert len(np.unique(group[np.r_[a,b]]))==len(a)*2 and np.all(fold[a]==fold[b])
        state=np.where(np.isnan(bills),2,(bills>0).astype(int))
        np.testing.assert_array_equal(state[a],state[b]); assert np.max(d)<=LIMIT+1e-12
        assert all(np.isfinite(p).all() for p in predictions.values())
        scores={'base':base,**predictions}
        npos=(bills[a]>0).sum(1)
        complete=np.isfinite(bills[a]).all(1)
        masks=dict(all=np.ones(len(a),bool), zero_complete=(npos==0)&complete,
                   zero_with_missing=(npos==0)&~complete, one_positive=npos==1, multi_positive=npos>=2,
                   strict_10pct=(d<=np.log(1.10)), hard_member=hard[a]|hard[b], common_member=wrong[a]|wrong[b])
        same_sleep = x.CryoSleep.to_numpy()[a] == x.CryoSleep.to_numpy()[b]
        masks.update(zero_same_sleep=masks['zero_complete'] & same_sleep,
                     multi_same_sleep=masks['multi_positive'] & same_sleep)
        report=summary(y,a,b,d,scores,masks)
        row_accuracy={name:dict(accuracy=float(((p>=.5)==y).mean()), hard466=int((((p>=.5)==y)&hard).sum()),
                 common1070=int((((p>=.5)==y)&wrong).sum())) for name,p in scores.items()}
        pair_labels_sec={}
        for name,score in scores.items():
            member=[]
            for left,right in ((a,b),(b,a)):
                ix=hard[left]&(y[left]!=y[right]);u=left[ix];v=right[ix]
                delta=(2*y[u].astype(int)-1)*(score[u]-score[v]);member.extend(np.where(abs(delta)<1e-12,.5,delta>0))
            pair_labels_sec[name]=dict(hard_members=len(member),concordant_credit=float(np.sum(member)))
        final[str(seed)]=dict(pairs=report, row_accuracy=row_accuracy, hard_member_outcomes=pair_labels_sec)
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=base,p=predictions['register'],
                            pair_left_row=row[a],pair_right_row=row[b],pair_max_log_gap=d,**predictions)
        print(seed,[(s,report[s]['pairs'],report[s]['discord'], {n:round(m['concordance'],3) if m['concordance'] is not None else None
                        for n,m in report[s]['scores'].items()}) for s in ['all','zero_complete','one_positive','multi_positive']],flush=True)
    write(ROOT/'reports/nov23_bill_matched_register/result.json',dict(results=final,limit=LIMIT))


if __name__=='__main__':
    with threadpool_limits(2):run()
