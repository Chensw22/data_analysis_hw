"""Test how far the composition of changed test passengers explains local-to-public reversals."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from broad_geographic_transfer_n23 import BILLS
from error_overlap_diagnostic import FAMILIES, load_authenticated
from export_consensus_routes import BASE, HISTORICAL
from export_nov21_error_batch import test_probabilities
from kaggle_workflow import sha256
from released_correction_audit_n23 import ART, ROOT, REPORT as RELEASE_REPORT, development, release, write
from robust_release_center_n23 import decisions


def context_fields(frame):
    bills = frame[BILLS]
    size = frame.PassengerId.str.split('_').str[0].map(frame.PassengerId.str.split('_').str[0].value_counts())
    # Compute size on the whole raw train/test partition, before subsetting development passengers.
    deck = frame.Cabin.str.split('/').str[0]
    context = np.c_[size.gt(1), size.ge(4), frame.Age.lt(15), frame.Age.ge(40), frame.Age.isna(),
                    frame.CryoSleep.eq(True), frame.CryoSleep.isna(), frame.HomePlanet.eq('Earth'),
                    frame.HomePlanet.eq('Europa'), deck.eq('F'), deck.eq('G'), frame.Cabin.isna(),
                    bills.gt(0).any(axis=1), bills.isna().any(axis=1), np.log1p(bills.sum(axis=1))/8]
    return context.astype(float)


def model_fields(anchor, other):
    return np.c_[anchor, abs(anchor-.5)*2, (other >= .5).mean(1), other.mean(1), other.std(1)]


def balanced(source, target):
    scale = source.std(0)
    keep = scale > .01
    if not keep.any():
        return np.full(len(source), 1/len(source))
    center = source.mean(0)[keep]
    x = ((source[:, keep]-center)/scale[keep]).clip(-4, 4)
    target_mean = ((target[:, keep]-center)/scale[keep]).clip(-4, 4).mean(0)
    def objective(v):
        tilted = x@v
        weights = softmax(tilted)
        return (logsumexp(tilted)-np.log(len(x))-target_mean@v+.5*np.dot(v, v),
                weights@x-target_mean+v)
    sol = minimize(objective, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    return softmax(x@sol.x)


def estimate(source, target, h, tfrac):
    xs, xt = source[0], target[0]
    if len(xs) == 0 or len(xt) == 0:
        return None
    score = {}
    directions = [np.c_[xs, source[1], source[2]], np.c_[xt, target[1], target[2]]]
    for k, weight in [('unweighted', np.ones(len(xs))/len(xs)),
                      ('direction', balanced(source[2], target[2])),
                      ('composition', balanced(*directions))]:
        score[k] = dict(forecast_gain_pp=float(100*tfrac*(weight@h)),
                        effective_rows=float(1/(weight@weight)))
    return score


def run():
    inp, _, folds, models, _ = load_authenticated()
    y, local, _ = development()
    raw = pd.read_csv(ROOT/'data/raw/train.csv')
    test = pd.read_csv(ROOT/'data/raw/test.csv')
    ids = test.PassengerId
    official = json.loads((RELEASE_REPORT/'primary_receipts.json').read_text())['scores']
    released = {k:release(k, ids, official) for k in local}
    dev = inp['dev']
    np.testing.assert_array_equal(raw.Transported.to_numpy()[dev], y)
    with np.load(ART/'v108_exact_oof/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'], dev)
        p = z['v108']
    pt = test_probabilities(BASE, ids)
    other = np.column_stack([models[k] for k in FAMILIES])
    ot = np.column_stack([test_probabilities(k, ids) for k in HISTORICAL])
    ctx, ct = context_fields(raw)[dev], context_fields(test)
    conf, cf = model_fields(p, other), model_fields(pt, ot)
    name = 'nov23_robust_release_center'
    local[name] = decisions(np.c_[other, p], p, local[131], folds)['rank_trim']
    rec, = (ROOT/'reports'/name/'online_attempts').glob('*.json')
    receipt = json.loads(rec.read_text())
    csv = ART/name/'submission.csv'
    assert receipt['evaluation']['Status'] == 'COMPLETE' and receipt['file_sha256'] == sha256(csv)
    submitted = pd.read_csv(csv)
    np.testing.assert_array_equal(submitted.PassengerId, ids)
    released[name] = (float(receipt['evaluation']['Public Score']), submitted.Transported.to_numpy())
    results = {}
    for n in [*range(114, 137), name]:
        anchor = 131 if n == name else 108 if n < 129 else 128
        b, bt = local[anchor], released[anchor][1]
        m, mt = local[n] != b, released[n][1] != bt
        source = (ctx[m], conf[m], b[m, None].astype(float))
        target = (ct[mt], cf[mt], bt[mt, None].astype(float))
        gain = np.where(local[n][m] == y[m], 1., -1.)
        results[str(n)] = dict(anchor=anchor, dev_changes=int(m.sum()), test_changes=int(mt.sum()),
                               local_pp=float(100*gain.sum()/len(y)),
                               actual_pp=round(100*(released[n][0]-released[anchor][0]), 5),
                               estimators=estimate(source, target, gain, mt.mean()))
    quality = {}
    for arm in ['local_pp', 'unweighted', 'direction', 'composition']:
        truth, pred = [], []
        for v in results.values():
            if v['estimators'] is None:
                continue
            truth.append(v['actual_pp'])
            pred.append(v['local_pp'] if arm == 'local_pp' else v['estimators'][arm]['forecast_gain_pp'])
        truth, pred = np.array(truth), np.array(pred)
        quality[arm] = dict(rows=len(pred), sign_agree=int((np.sign(truth) == np.sign(pred)).sum()),
                            mean_abs_error_pp=float(abs(truth-pred).mean()), spearman=float(spearmanr(truth, pred).statistic))
    write(ROOT/'reports/nov23_bill_value/public_transfer.json', dict(releases=results, quality=quality))
    print(quality, results[name], flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        run()
