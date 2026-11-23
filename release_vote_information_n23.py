"""Measure how many labels the four most recent submission recipes can distinguish."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from broad_booking_policy_replay_n23 import interval
from broad_geographic_transfer_n23 import load
from broad_pairwise_adjudication_n23 import paired
from broad_partition_evidence_n23 import ART, ROOT, write
from broad_refit_dissent_n23 import features
from error_overlap_diagnostic import FAMILIES, load_authenticated
from kaggle_workflow import sha256
from robust_release_center_n23 import decisions
from specialist_full_route_n23 import decision, routing_partitions, use_tree
from specialist_v131_transfer_n23 import read

NAMES = ['incumbent', 'six_models', 'budget_tree', 'ten_experts', 'budget_tabicl']
DIRS = ['proposal_v131_party_consensus', 'nov23_robust_release_center',
        'proposal_n23_budget_quarter_three_family', 'nov23_specialist_full_manifest',
        'proposal_n23_incumbent_quarter_tabicl_budget']
REPORT = ROOT/'reports/nov23_release_vote_information'


def encoding(votes):
    return votes.astype(int) @ (2**np.arange(votes.shape[1]))


def counts(votes, y):
    bits = encoding(votes)
    space = 2**votes.shape[1]
    yes = np.bincount(bits, weights=y, minlength=space).astype(int)
    n = np.bincount(bits, minlength=space)
    return n, yes


def bookend(votes, y):
    n, yes = counts(votes, y)
    hits = votes == y[:, None]
    return dict(rows=len(y), all_wrong=int((~hits.any(1)).sum()), any_right=int(hits.any(1).sum()),
                best_after_seeing_labels=int(np.maximum(yes, n-yes).sum()), unique_patterns=int((n>0).sum()),
                winner_only=[int(((hits.sum(1)==1)&hits[:,j]).sum()) for j in range(votes.shape[1])])


def field_correction(votes, y, t, q, raw, continuous, include_fields):
    n, yes = counts(votes[t], y[t])
    codes = encoding(votes)
    # Small cells shrink toward the fraction of opinions voting yes.
    prior = (1+votes.sum(1))/(2+votes.shape[1])
    local = (yes[codes]+10*prior)/(n[codes]+10)
    if not include_fields:
        return local[q] >= .5
    # Use only label-independent vote inputs for the logistic regression;
    # cell frequencies above are used solely by the independent pattern comparator.
    design = np.c_[votes, continuous, raw]
    scale = StandardScaler().fit(design[t])
    model = LogisticRegression(C=.04, max_iter=500).fit(scale.transform(design[t]).clip(-6, 6), y[t])
    return model.predict(scale.transform(design[q]).clip(-6, 6))


def main():
    row, y, groups, fold, v80, v108, v131, cold, experts = read()
    rows, truth, frame, _, _ = load()
    np.testing.assert_array_equal(row, rows)
    np.testing.assert_array_equal(y, truth)
    _, _, raw_fold, members, _ = load_authenticated()
    np.testing.assert_array_equal(fold, raw_fold)
    six = decisions(np.column_stack([members[k] for k in FAMILIES]+[v108]), v108, v131, fold)['rank_trim']
    with np.load(ART/'nov23_budget_quarter_replay/original.npz') as z:
        np.testing.assert_array_equal(z['row'], row)
        np.testing.assert_array_equal(z['fold'], fold)
        cat = z['alternative']
    with np.load(ART/'nov23_pretrained_spending_views/v131.npz') as z:
        np.testing.assert_array_equal(z['row'], row)
        np.testing.assert_array_equal(z['fold'], fold)
        tab = z['budget']
    tree = (.75*v80+.25*cat)>=.5
    locked = (v108>=.5)!=v131
    foundation = np.where(locked, v131, (.75*v108+.25*tab)>=.5)
    for vote, expected in [(six,(116,86)),(tree,(204,170))]:
        result = paired(vote==y,v131==y)
        assert (result['fixed'],result['harmed']) == expected
    raw = features(frame)[0]
    continuous = logit(np.c_[v80, v108, cat, tab].clip(.003,.997))
    x = np.c_[v80, experts]
    histories, examples, split_checks = {}, [], []
    for seed in [481,179]:
        methods = {k:v131.copy() for k in ['majority','patterns','with_fields']}
        paper = np.zeros((len(y),5),bool)
        visited = np.zeros(len(y),int)
        for f in np.unique(fold):
            ix = np.flatnonzero(fold==f)
            splits = np.zeros(len(ix),int)
            for s, (_, b) in enumerate(GroupKFold(5, shuffle=True, random_state=seed).split(ix, groups=groups[ix])):
                splits[b] = s
            parts = routing_partitions(x[ix])
            for s in range(5):
                q = np.flatnonzero(splits==s)
                t = np.flatnonzero(np.isin(splits,[(s+1)%5,(s+2)%5]))
                cal = np.flatnonzero(~np.isin(splits,[s,(s+1)%5,(s+2)%5]))
                assert all(not np.intersect1d(groups[ix[a]], groups[ix[b]]).size
                           for a,b in [(cal,t),(cal,q),(t,q)])
                # The ten-expert calibration excludes both the combination training and evaluation bookings.
                out = np.r_[t,q]
                p, _ = use_tree(parts, x[ix], cal, out, y[ix[cal]])
                who = ix[out]
                ten = decision(p, v80[who])
                votes = np.column_stack([v131[who], six[who], tree[who], ten, foundation[who]])
                tr = np.arange(len(t))
                va = np.arange(len(t),len(out))
                paper[ix[q]] = votes[va]
                visited[ix[q]] += 1
                methods['majority'][ix[q]] = votes[va].sum(1)>=3
                for label in ['patterns','with_fields']:
                    methods[label][ix[q]] = field_correction(votes, y[who], tr, va, raw[who], continuous[who],
                                                             label=='with_fields')
        assert (visited==1).all()
        examples.append(paper)
        histories[str(seed)] = {}
        for k,v in methods.items():
            hit = v==y
            gain = hit.astype(float)-(v131==y)
            histories[str(seed)][k]=dict(**paired(hit,v131==y), gain_ci95_pp=interval(gain,groups),
                                          cold=paired(hit[cold],v131[cold]==y[cold]),
                                          by_fold=[int(gain[fold==f].sum()) for f in np.unique(fold)])
        split_checks.append(bookend(paper,y))
    # Every bit in the official test pattern is verified against its actual submission receipt.
    test = pd.read_csv(ROOT/'data/raw/test.csv', usecols=['PassengerId'])
    test_bank = []
    for name in DIRS:
        csv=ART/name/'submission.csv'
        entry,=(ROOT/'reports'/name/'online_attempts').glob('*.json')
        rec=json.loads(entry.read_text())
        assert rec['evaluation']['Status']=='COMPLETE' and rec['file_sha256']==sha256(csv)
        b=pd.read_csv(csv)
        np.testing.assert_array_equal(b.PassengerId,test.PassengerId)
        test_bank.append(b.Transported.to_numpy())
    official = np.column_stack(test_bank)
    cells, yes = counts(examples[0],y)
    celltest = np.bincount(encoding(official),minlength=32)
    fixed_wrong = np.logical_and.reduce([(~(v==y[:,None]).any(1)) for v in examples])
    historical = []
    for s in [20261112,20261115,20261116]:
        with np.load(ART/f'nov23_group_kernel/{s}.npz') as z:
            np.testing.assert_array_equal(z['row'],row)
            historical.append((z['base']>=.5)!=y)
    wrong_count=np.sum(historical,axis=0)
    subsets={str(i):bookend(examples[0][wrong_count==i], y[wrong_count==i]) for i in range(4)}
    report=dict(names=NAMES, population=split_checks, by_historical_error_count=subsets,
                still_wrong_both=int(fixed_wrong.sum()), methods=histories,
                vote_patterns=[dict(code=i,dev=int(cells[i]),dev_true=int(yes[i]),test=int(celltest[i])) for i in range(32)])
    write(REPORT/'panel.json', report)
    outdir = ART/'nov23_release_vote_information'
    outdir.mkdir(exist_ok=True)
    np.savez_compressed(outdir/'panel.npz',row=row,y=y,fold=fold,groups=groups,votes=np.stack(examples),official=official)
    print(split_checks,histories,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):
        main()
