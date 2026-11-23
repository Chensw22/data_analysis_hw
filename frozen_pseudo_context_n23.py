"""Measure the effect of adding real, self-labeled or shuffled context on identical bookings."""
import hashlib

import numpy as np
from scipy.special import logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from broad_bill_exception_n23 import strata
from broad_boosting_path_n23 import calibrated
from broad_geographic_transfer_n23 import load
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, historical, write
from broad_release_miss_coverage_n23 import summarize
from tabicl_routed_context_n23 import DIGEST, WEIGHT, infer, torch


def contexts(group, train, seed):
    unique, counts = np.unique(group[train], return_counts=True)
    order = np.random.default_rng(seed).permutation(len(counts))
    ends = counts[order].cumsum()
    cut, end = [int(np.argmin(abs(ends - n))) + 1 for n in (1024, 1536)]
    first = train[np.isin(group[train], unique[order[:cut]])]
    extra = train[np.isin(group[train], unique[order[cut:end]])]
    return first, extra


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == DIGEST
    row, y, frame, group, _ = load()
    ids, target, masks = historical()
    np.testing.assert_array_equal(ids, row)
    np.testing.assert_array_equal(target, y)
    state = strata(frame)
    masks.update(paid=masks['repeat'] & (state == 2) & y, zero=masks['repeat'] & (state == 0) & ~y)
    out = ART / 'nov23_frozen_pseudo_context'
    out.mkdir(exist_ok=True)
    hist, references, details = {}, [], {}
    for seed in SEEDS:
        with np.load(ART / f'nov23_group_kernel/{seed}.npz') as record:
            np.testing.assert_array_equal(record['row'], row)
            np.testing.assert_array_equal(record['y'], y)
            base, fold = record['base'], record['fold']
        scores = {k: np.zeros(len(y)) for k in ['original', 'gold', 'pseudo', 'shuffled']}
        additions = []
        for f in np.unique(fold):
            q = np.flatnonzero(fold == f)
            a, b = contexts(group, np.flatnonzero(fold != f), seed + int(f))
            for left, right in [(a, b), (a, q), (b, q)]:
                assert not np.intersect1d(group[left], group[right]).size
            model_seed = 681 + int(f)
            # The teacher observes only the original support labels, including when labeling b.
            teacher = infer(frame, y, a, np.r_[b, q], model_seed)
            pseudo = teacher[:len(b)] >= .5
            scores['original'][q] = teacher[len(b):]
            support = np.sort(np.r_[a,b])
            for name, extension in [('gold',y[b]), ('pseudo',pseudo),
                                    ('shuffled',np.random.default_rng(531+int(f)).permutation(pseudo))]:
                labels = np.zeros_like(y)
                labels[a] = y[a]
                labels[b] = extension
                scores[name][q] = infer(frame, labels, support, q, model_seed)
            additions.append(dict(initial=len(a), added=len(b), correct_pseudo=int((pseudo == y[b]).sum()),
                                  validation=len(q)))
        decisions = {name: p >= .5 for name,p in scores.items()}
        for name,p in scores.items():
            x = logit(np.column_stack([base,p]).clip(.002,.998))
            vote = np.zeros(len(row),dtype=bool)
            for f in np.unique(fold):
                ix = np.flatnonzero(fold == f)
                for a,b in GroupKFold(3,shuffle=True,random_state=925).split(ix,groups=group[ix]):
                    t,q = ix[a],ix[b]
                    assert not np.intersect1d(group[t],group[q]).size
                    vote[q] = calibrated(x[t],x[q],y,t) >= .5
            decisions['combined_'+name] = vote
        references.append((base >= .5) == y)
        changes = {}
        for name,vote in decisions.items():
            h = vote == y
            hist.setdefault(name,[]).append(h)
            previous = decisions['combined_original' if name.startswith('combined_') else 'original'] == y
            changes[name] = dict(fixed=int((h & ~previous).sum()), harmed=int((~h & previous).sum()))
        np.savez_compressed(out/f'{seed}.npz',row=row,y=y,fold=fold,base=base,**scores)
        details[str(seed)] = dict(folds=additions, vs_original=changes)
        print(seed,changes['gold'],changes['pseudo'],changes['shuffled'],flush=True)
    write(ROOT/'reports/nov23_frozen_pseudo_context/results.json',dict(plans=details, grouped={
        k:summarize(np.stack(h),np.stack(references),masks) for k,h in hist.items()}))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
