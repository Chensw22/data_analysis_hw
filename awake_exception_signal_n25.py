"""Test whether integer spending patterns identify awake transported passengers at the same spending scale."""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from disjoint_tree_fields_n25 import ART, RELEASE, OUTER, DEST
from full_roster_conditional_pool_n23 import BILLS, data
from full_roster_six_failure_archive_n24 import prediction_sets


def categories(raw, g):
    fees = raw[BILLS].to_numpy(float)
    positive = fees > 0
    value = np.where(positive, fees, 0).astype(int)
    cab = raw.Cabin.str.split('/', expand=True)
    frame = raw[['HomePlanet', 'Destination', 'VIP']].copy()
    frame['deck'], frame['side'] = cab[0], cab[2]
    frame['age'] = np.floor(raw.Age/5)
    frame['order_size'] = pd.Series(g).map(pd.Series(g).value_counts()).clip(upper=5)
    frame['total'] = np.floor(4*np.log1p(value.sum(1)))
    frame['mask'] = positive.astype(int) @ (2**np.arange(value.shape[1]))
    frame['planet_by_mask'] = frame.HomePlanet.astype(str)+'/'+frame['mask'].astype(str)
    frame['deck_by_side'] = frame.deck.astype(str)+'/'+frame.side.astype(str)
    frame['positive_services'] = positive.sum(1)
    digits, bins = [], []
    for j, service in enumerate(BILLS):
        v = value[:,j]
        mag = np.floor(2*np.log2(1+v)).astype(int)
        frame[service+'_scale'] = np.where(np.isnan(fees[:,j]), -1, mag)
        frame[service+'_share'] = np.floor(8*v/np.maximum(1,value.sum(1))).astype(int)
        # Digit 0/5 is more likely when amounts are rounded, with further tiers for 10, 50 and 100.
        tier = np.select([v==0, v%100==0, v%50==0, v%10==0, v%5==0], [-1,4,3,2,1], default=0)
        digits.append(np.column_stack([np.where(v>0, v%10, -1), tier]))
        bins.append(frame[service+'_scale'].to_numpy())
    return frame.astype(str), digits, bins


def permute_within_scale(digits, bins, folds):
    rng = np.random.default_rng(613)
    out = []
    for values, scales in zip(digits,bins,strict=True):
        shuffled = values.copy()
        # Keep distributions fixed inside both the amount bin and the outer fold.
        for f in np.unique(folds):
            for scale in np.unique(scales[folds==f]):
                k = np.flatnonzero((folds==f) & (scales==scale))
                shuffled[k] = values[rng.permutation(k)]
        out.append(shuffled)
    return out


def extra(digits, bins):
    frame = {}
    for service,digit,scale in zip(BILLS,digits,bins,strict=True):
        frame[service+'_last'] = digit[:,0]
        frame[service+'_round'] = digit[:,1]
        frame[service+'_round_at_scale'] = (scale//4+1)*6 + digit[:,1]
    all_tiers = np.stack([v[:,1] for v in digits],axis=1)
    for tier in (1,2,3,4):
        frame['number_at_tier_'+str(tier)] = (all_tiers>=tier).sum(1)
    return pd.DataFrame(frame).astype(str)


def effects(pred, ref, truth, mask, fold):
    fixed = (pred==truth)&(ref!=truth)&mask
    harmed = (pred!=truth)&(ref==truth)&mask
    return dict(rows=int(mask.sum()), fixed=int(fixed.sum()), harmed=int(harmed.sum()),
                by_fold=[int(fixed[fold==k].sum()-harmed[fold==k].sum()) for k in range(8)])


def run():
    raw,y,g,f = data()
    awake = raw.CryoSleep.eq(False).to_numpy(bool)
    control, digits, bins = categories(raw,g)
    # Shuffle within awake status too, so the placebo targets the awake-only fitting population.
    placebo = permute_within_scale(digits,bins,2*f+awake.astype(int))
    columns = [control, extra(digits,bins), extra(placebo,bins)]
    modes = ['amount_scale', 'rounded_amounts', 'shuffled_endings']
    prob = np.full((len(y),len(modes)),np.nan)
    for k in range(8):
        t = np.flatnonzero((f!=k)&awake)
        q = np.flatnonzero((f==k)&awake)
        assert not np.isin(g[t],g[q]).any()
        parts = []
        for x in columns:
            enc = OneHotEncoder(handle_unknown='ignore')
            parts.append((enc.fit_transform(x.iloc[t]),enc.transform(x.iloc[q])))
        for j in range(len(modes)):
            idx = [0] if j==0 else [0,j]
            train, query = [hstack([parts[i][side] for i in idx],format='csr') for side in (0,1)]
            clf = LogisticRegression(C=.2,solver='liblinear',max_iter=250,random_state=683)
            clf.fit(train,y[t])
            prob[q,j] = clf.predict_proba(query)[:,1]
    assert np.isfinite(prob[awake]).all()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _, panels=prediction_sets(raw,y,g,f,rows,hf)
    archive = np.column_stack(panels[0])>=.5
    with np.load(ART/'nov25_production_anchor_cpu/oof.npz') as z:
        np.testing.assert_array_equal(z['fold'],f)
        p, known, nine = [z[key].copy() for key in ('v96','certified_v131_equals_v96','certified_nine')]
    for k in OUTER:
        with np.load(RELEASE/f'release{k}.npz') as z:
            q=z['q']
            np.testing.assert_array_equal(q,np.flatnonzero(f==k))
            known[q],nine[q] = z['v131']>=0,z['nine']
    native = p>=.5
    unanimous = (archive==archive[:,:1]).all(1)
    archive_ref = archive[:,0]
    report = {}
    for j, mode in enumerate(modes):
        for rule in (['direct','quarter_blend'] if j==0 else ['direct','quarter_blend','add_digit_log_odds']):
            if rule=='direct':
                alt=prob[:,j]
            elif rule=='quarter_blend':
                alt=.75*p+.25*prob[:,j]
            else:
                alt=expit(logit(np.clip(p,1e-6,1-1e-6))+logit(prob[:,j])-logit(prob[:,0]))
            chosen=np.where(awake,alt>=.5,native)
            changes=chosen!=native
            # A candidate may only change known nine predictions when its direction agrees with native beforehand.
            nine_alt=np.where(changes & (nine==native),chosen,nine)
            changes_archive=awake & (native==archive_ref) & changes
            arch_alt=np.where(changes_archive,chosen,archive_ref)
            only_negative = archive_ref==False
            report[mode+'/'+rule]=dict(
                all_native=effects(chosen,native,y,np.ones(len(y),bool),f),
                awake_native=effects(chosen,native,y,awake,f),
                known_nine=effects(nine_alt,nine,y,known,f),
                unseen_order_nine=effects(nine_alt,nine,y,known & ~np.isin(g,g[rows]),f),
                awake_unanimous=effects(arch_alt,archive_ref,y,unanimous & awake,f),
                awake_unanimous_predicted_negative=effects(arch_alt,archive_ref,y,unanimous & awake & only_negative,f))
    result=dict(awake=int(awake.sum()),all_rows=len(y),exact_nine=int(known.sum()),models=report,
                note='Digit shuffles preserve each service amount bin, awake status, and outer fold. All rules keep non-awake predictions.')
    DEST.with_name('awake_exception_signal_n25.json').write_text(json.dumps(result,indent=2)+'\n')
    for k,v in report.items():
        print(k,[(v[n]['fixed'],v[n]['harmed']) for n in ('all_native','known_nine','unseen_order_nine','awake_unanimous','awake_unanimous_predicted_negative')])

if __name__=='__main__':
    run()
