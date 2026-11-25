"""Score a held-out manifest field under both outcome hypotheses, with and without precise bills."""
import argparse
import json

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from full_roster_booking_validation_n23 import ROOT, data

OUT = ROOT/'artifacts/nov25_conditional_manifest_evidence'
PRECISION = ROOT/'artifacts/nov25_within_bin_bill_precision'
REPORT = ROOT/'reports/nov23_matched_learning_curve/conditional_manifest_evidence.json'


def field_scores(model, query, position, field, target):
    import torch
    n = len(query)
    both = query.repeat(2, 1)
    both[:, target] = torch.arange(2).repeat_interleave(n)
    mask = torch.ones_like(both, dtype=torch.bool)
    mask[:, field] = False
    scores = model(both, mask, position.repeat(2, 1))[:, field].log_softmax(1)
    likelihood = scores.gather(1, both[:, field, None]).ravel()
    return (likelihood[n:] - likelihood[:n]).numpy()


def generate():
    import torch
    from passenger_joint_likelihood_n25 import TARGET, encode, train_model
    from within_bin_bill_precision_n25 import coordinates, predictions
    torch.set_num_threads(2)
    raw, y, group, fold = data()
    OUT.mkdir(exist_ok=True)
    for k in range(8):
        q, t = np.flatnonzero(fold==k), np.flatnonzero(fold!=k)
        assert not np.isin(group[t], group[q]).any()
        path = OUT/f'fold{k}.npz'
        if path.exists():
            continue
        enc, cards = encode(raw, t)
        pos, _ = coordinates(raw, enc, t, q, k)
        x = torch.as_tensor(np.column_stack([enc, y.astype(int)]))
        query = x[q].clone()
        query[:, TARGET] = 0
        exact = train_model(x, cards, t, 1241+k, True, pos)
        with np.load(PRECISION/f'{k}_exact_bill_position.npz') as z:
            for name, p in predictions(exact, query, pos[q]).items():
                np.testing.assert_array_equal(p, z[name])
        with torch.inference_mode():
            outputs = {}
            for name, coordinate in [('observed', pos[q]), ('neutral', torch.zeros_like(pos[q]))]:
                outputs[name] = np.column_stack([field_scores(exact, query, coordinate, j, TARGET)
                                                  for j in range(TARGET)])
                assert np.isfinite(outputs[name]).all()
                # The two outcome hypotheses overwrite every query's placeholder.
                other = query[:20].clone()
                other[:, TARGET] = 1
                check = field_scores(exact, other, coordinate[:20], 0, TARGET)
                np.testing.assert_allclose(check, outputs[name][:20, 0], atol=1e-5, rtol=0)
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as buf:
            np.savez_compressed(buf, q=q, t=t, train_y=y[t], passenger=raw.PassengerId.iloc[q].to_numpy(str), **outputs)
        tmp.replace(path)
        print('field evidence', k, flush=True)


def evaluate():
    from adjudicate_amount_dissent_n25 import counts, load_fold
    from disjoint_tree_fields_n25 import ART
    from full_roster_six_failure_archive_n24 import prediction_sets
    from grouped_peer_release_replay_n25 import edit_effect_bounds
    raw, y, group, fold = data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows, hf = z['rows'], z['fold']
    _, panels = prediction_sets(raw, y, group, fold, rows, hf)
    votes, old = [(np.column_stack(p)>=.5).sum(1) for p in panels]
    common = (votes==0)|(votes==68)
    ref68 = votes>34
    repeated = np.zeros(len(y), bool)
    repeated[rows] = common[rows]&((old==0)|(old==68))&((old>34)==ref68[rows])
    new = ~np.isin(group, group[rows])
    records = {}
    for k in range(8):
        q, t = np.flatnonzero(fold==k), np.flatnonzero(fold!=k)
        ref, prior, low, high, controls, _, _ = load_fold(k, q, t, raw, y)
        with np.load(OUT/f'fold{k}.npz') as z:
            for key, expected in [('t',t), ('q',q), ('train_y',y[t]), ('passenger',raw.PassengerId.iloc[q])]:
                np.testing.assert_array_equal(z[key], expected)
            exact, neutral = z['observed'], z['neutral']
        sign = np.where(ref,-1.,1.)[:,None]
        # These are overlapping conditional tests, so their coefficients are learned together.
        # Train on every booking: a strong likelihood ratio may overturn even an initially unanimous decision.
        matrices = {'prior':controls['anchor'], 'midpoint':np.column_stack([controls['anchor'], sign*neutral]),
                    'observed':np.column_stack([controls['anchor'], sign*exact]),
                    'precision_difference':np.column_stack([controls['anchor'], sign*(exact-neutral)]),
                    'joint':np.column_stack([controls['anchor'],sign*neutral,sign*(exact-neutral)])}
        designs = {'all':list(GroupKFold(4,shuffle=True,random_state=3207).split(q,groups=group[q])),
                   'new':[(np.flatnonzero(~new[q]), np.flatnonzero(new[q]))]}
        truth = y[q]
        for design, splits in designs.items():
            for train, test in splits:
                assert not np.isin(group[q[train]],group[q[test]]).any()
                for name, matrix in matrices.items():
                    model = LogisticRegression(C=.2,solver='liblinear',max_iter=500)
                    model.fit(matrix[train], ref[train]!=truth[train])
                    change = model.predict_proba(matrix[test])[:,1]>.5
                    a,b = edit_effect_bounds(ref[test]^change,ref[test],prior[test],low[test],high[test],truth[test])
                    known = prior[test]>=0
                    np.testing.assert_array_equal(a[known], b[known])
                    records.setdefault((design,name),[]).append((q[test],known,a,b))
    report = {}
    for (design,name), parts in records.items():
        ix, known, a, b = (np.concatenate([part[j] for part in parts]) for j in range(4))
        np.testing.assert_array_equal(np.sort(ix), np.flatnonzero(new) if design=='new' else np.arange(len(y)))
        r = dict(bounds=[int(a.sum()),int(b.sum())])
        for key,scope in [('known',known),('unanimous',known&common[ix]),('repeated',known&repeated[ix]),
                          ('shared_errors',known&common[ix]&(ref68[ix]!=y[ix]))]:
            r[key]=counts(a,scope,fold[ix])
        report.setdefault(design,{})[name]=r
        print(design,name,r['bounds'],r['known']['fixed'],r['known']['harmed'],r['shared_errors']['fixed'])
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['generate','evaluate'])
    generate() if p.parse_args().action=='generate' else evaluate()
