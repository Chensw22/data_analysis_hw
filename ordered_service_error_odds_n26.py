"""Calibrate actual release error odds by the order of the passenger's service charges."""
import json
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from full_roster_booking_validation_n23 import ROOT
from release_field_contributions_n26 import DECK, FEES, OUT, REVIEW, evidence, measures, outcome_scores, panel_predictions, protocols, replay, sources


def signatures(raw):
    amount = raw[FEES].fillna(-1).to_numpy()
    order = np.argsort(-amount, axis=1, kind='stable')
    values = np.take_along_axis(amount, order, axis=1)
    ranked = np.where(values[:, :2] > 0, order[:, :2], -1)
    # Give an equal positive amount its own tie category, independent of column order.
    for j in (0, 1):
        tie = values[:, j] == values[:, j+1]
        if j:
            tie |= values[:, j] == values[:, j-1]
        ranked[(values[:, j] > 0) & tie, j] = 5
    return np.column_stack([(amount > 0).sum(1), (amount < 0).sum(1), ranked])


def prior(xt, xq, target, groups):
    def predict(t, x):
        scaler = StandardScaler().fit(xt[t])
        fit = LogisticRegression(C=.25, max_iter=500).fit(scaler.transform(xt[t]), target[t])
        return fit.decision_function(scaler.transform(x)).clip(-8, 8)
    oof = np.empty(len(xt))
    for t, q in GroupKFold(3, shuffle=True, random_state=829).split(xt, groups=groups):
        assert not np.isin(groups[t], groups[q]).any()
        oof[q] = predict(t, xt[q])
    return oof, predict(np.arange(len(xt)), xq)


def adjust(train, query, yt, train_logit, query_logit):
    # A child estimates an additional intercept after applying its parent's correction.
    # Penalization gives sparse or unobserved cells nearly the parent odds.
    t, q = train_logit.copy(), query_logit.copy()
    pred = [expit(q)]
    for columns in ([0,1,2,3], [0,1,2,3,4], [0,1,2,3,4,5]):
        keys, inverse = np.unique(np.vstack([train[:,columns],query[:,columns]]),axis=0,return_inverse=True)
        ti, qi = inverse[:len(t)], inverse[len(t):]
        delta = np.zeros(len(keys))
        for _ in range(12):
            prob = expit(t + delta[ti])
            grad = np.bincount(ti, weights=yt-prob, minlength=len(keys)) - 5*delta
            info = np.bincount(ti, weights=prob*(1-prob), minlength=len(keys)) + 5
            delta += grad/info
        t += delta[ti]
        q += delta[qi]
        pred.append(expit(q))
    return pred


def placebo(matrix, logits, rng):
    result = matrix.copy()
    # Shuffle service identities together; retain direction, consensus, missingness, activity and risk band.
    keys = np.column_stack([matrix[:,:4], np.digitize(expit(logits), [.1,.25,.4,.5,.65])])
    _, group = np.unique(keys, axis=0, return_inverse=True)
    for k in np.unique(group):
        ids = np.flatnonzero(group == k)
        result[ids,4:] = matrix[rng.permutation(ids),4:]
    return result


def run():
    raw,y,g,panels = sources()
    bill = signatures(raw)
    panel = {tag:panel_predictions(v,tag) for tag,v in panels.items()}
    scores = {tag:outcome_scores(v,panel[tag]) for tag,v in panels.items()}
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    bags = {}
    for key,src,dst,t,q in protocols(panels,g):
        a,b = src['ids'][t],dst['ids'][q]
        assert not np.isin(g[a],g[b]).any()
        with np.load(OUT/f'{key}.npz') as z:
            np.testing.assert_array_equal(z['ids'],a)
            np.testing.assert_array_equal(z['outer'],b)
            sr = z['risk']
        with np.load(REVIEW/f'pred_{key}.npz') as z:
            np.testing.assert_array_equal(z['q'],b)
            qr = np.column_stack([z['identities_error'],z['identities_outcome']])
        old, ca, xt = evidence(src,scores['historical'],panel['historical'],t,sr,enabled[a])
        cur, cb, xq = evidence(dst,scores['full'],panel['full'],q,qr,enabled[b])
        np.testing.assert_array_equal(cur,replay(key,q,dst,panel['full'],enabled))
        train = np.column_stack([old,ca,bill[a]])
        query = np.column_stack([cur,cb,bill[b]])
        target = old!=y[a]
        ltr,lq = prior(xt,xq,target,g[a])
        actual = adjust(train,query,target,ltr,lq)
        rng = np.random.default_rng(1449)
        fake = adjust(placebo(train,ltr,rng),placebo(query,lq,rng),target,ltr,lq)
        np.testing.assert_allclose(actual[1],fake[1])
        tag = 'bookingmix' if key.startswith('bookingmix') else key
        bags.setdefault(tag,[]).append((b,cb,cur!=y[b],np.column_stack([*actual,*fake[2:]])))
    report = {}
    for tag,items in bags.items():
        ids,agree,wrong,p = [np.concatenate(v) for v in zip(*items,strict=True)]
        assert len(np.unique(ids))==len(ids)
        r = {}
        for s,m in [('all',np.ones(len(ids),bool)),('unanimous',agree),('contested',~agree)]:
            rows = {}
            for j,name in enumerate(['risk','counts','first','first_and_second','placebo_first','placebo_both']):
                active = m & (p[:,j]>.5) & (p[:,1]<=.5)
                retired = m & (p[:,j]<=.5) & (p[:,1]>.5)
                rows[name] = dict(**measures(p[m,j],wrong[m],ids[m]),extra_fixed=int((active&wrong).sum()),
                                  extra_harmed=int((active&~wrong).sum()), retired_fixes=int((retired&wrong).sum()),
                                  retired_harms=int((retired&~wrong).sum()))
            r[s] = rows
        service = bill[ids,2]
        r['unanimous_by_largest_service'] = {name:dict(rows=int(m.sum()),errors=int((m&wrong).sum()))
            for j,name in enumerate([*FEES,'equal_largest','none']) if (m:=agree&(service==(j if j<6 else -1))).any()}
        report[tag] = r
        print(tag, r['unanimous_by_largest_service'], r['all']['first_and_second'],flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/ordered_service_error_odds.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
