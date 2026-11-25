"""Test whether independent bookings before and after a berth share outcome residuals."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from solo_archive_recovery_n26 import ticker
from solo_selection_likelihood_n26 import fit_model

OUT = ROOT/'artifacts/nov26_bracketing_cabin_orders'
REPORT = ROOT/'reports/nov23_matched_learning_curve/bracketing_cabin_orders.json'
NAMES = ['passenger_only','real_neighbors','shuffled_neighbors']


def bracket(source, ids, coords, paid, lanes, residual_key):
    columns = []
    for by_spending in (False,True):
        block = np.zeros((len(ids),8))
        target_keys = np.array([f'{lane}:{int(spend)}' if by_spending else lane for lane,spend in zip(lanes[ids],paid[ids])])
        bank = source.copy()
        bank['key'] = bank.lane + ':' + bank.paid.astype(int).astype(str) if by_spending else bank.lane
        # Collapse repeated appearances of the same booking in this field to a single observation.
        bank = bank.groupby(['key','group'],as_index=False).agg(pos=('pos','median'),value=(residual_key,'mean'))
        for lane, rows in bank.groupby('key'):
            rows = rows.sort_values('pos')
            ix = np.flatnonzero((target_keys==lane)&np.isfinite(coords[ids]))
            if not len(ix):
                continue
            pos = rows.pos.to_numpy()
            agg = np.r_[0,np.cumsum(rows.value)]
            x = coords[ids[ix]]
            for k,width in enumerate((12,48)):
                a,left,right,b = [np.searchsorted(pos,z,side=s) for z,s in
                                  [(x-width,'left'),(x,'left'),(x,'right'),(x+width,'right')]]
                pre = (agg[left]-agg[a])/(3+left-a)
                post = (agg[b]-agg[right])/(3+b-right)
                # Same coordinate is counted separately. A signed minimum requires support on both sides.
                shared = np.where(pre*post>0,np.sign(pre)*np.minimum(abs(pre),abs(post)),0.)
                center = (agg[right]-agg[left])/(3+right-left)
                block[ix,k*4:(k+1)*4] = np.column_stack([pre+post,pre-post,shared,center])
        columns.append(block)
    return np.column_stack(columns)


def fit_fold(x, y, g, paid, lanes, coord, train, query, seed):
    parts = [train[q] for _,q in GroupKFold(3,shuffle=True,random_state=seed).split(train,groups=g[train])]
    for k,rows in enumerate([*parts,query]):
        others = np.concatenate([*parts,query][:k]+[*parts,query][k+1:])
        assert not np.isin(g[rows],g[others]).any()
    scores = []
    for r in range(3):
        fit,bank,cal = [parts[(r+i)%3] for i in range(3)]
        clf = fit_model(x.iloc[fit], y[fit])
        ids = np.r_[bank,cal,query]
        p = clf.predict_proba(x.iloc[ids])[:,1].clip(.002,.998)
        pb = p[:len(bank)]
        source = pd.DataFrame(dict(lane=lanes[bank],group=g[bank],pos=coord[bank],paid=paid[bank],
                                   risk=pb,residual=y[bank]-pb)).dropna(subset=['pos'])
        # Shuffle complete order-level residuals within lane, spending status, risk band and broad cabin section.
        source = source.groupby(['lane','paid','group'],as_index=False).agg(
            pos=('pos','median'),risk=('risk','mean'),residual=('residual','mean'))
        source['section'] = source.pos//160
        source['risk_band'] = np.digitize(source.risk,[.33,.67])
        rng = np.random.default_rng(seed+19*r)
        source['shuffled'] = source.groupby(['lane','paid','section','risk_band']).residual.transform(
            lambda s:rng.permutation(s.to_numpy()))
        target_ids = np.r_[cal,query]
        odds = logit(p[len(bank):])
        base = np.column_stack([odds,paid[target_ids],odds*paid[target_ids], x.Deck.iloc[target_ids]=='G'])
        pred = []
        for name in NAMES:
            design = base
            if name != 'passenger_only':
                local = bracket(source,target_ids,coord,paid,lanes,'residual' if name=='real_neighbors' else 'shuffled')
                design = np.column_stack([base,local,local*(~paid[target_ids,None]),local*base[:,3:4]])
            model = LogisticRegression(C=.8,max_iter=450).fit(design[:len(cal)],y[cal])
            pred.append(model.predict_proba(design[len(cal):])[:,1])
        scores.append(pred)
    return np.array(scores)


def run():
    raw,y,g,fold,views,votes = projection(include_full=True)
    x = compact_features(raw)
    x['booking_size'] = raw.PassengerId.groupby(g).transform('size')
    paid = raw[FEES].gt(0).any(axis=1).to_numpy()
    lanes = (x.Deck+'/'+x.Side).to_numpy()
    coord = x.CabinNumber.to_numpy()
    OUT.mkdir(exist_ok=True)
    score = np.full((3,3,len(y)),np.nan)
    for k in np.unique(fold):
        train,query = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        path = OUT/f'fold{k}.npz'
        if not path.exists():
            values = fit_fold(x,y,g,paid,lanes,coord,train,query,1171+int(k))
            with path.with_suffix('.tmp').open('wb') as h:
                np.savez_compressed(h,t=train,q=query,y=y[train],passenger=raw.PassengerId.iloc[query].to_numpy(str),p=values)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for key,v in [('t',train),('q',query),('y',y[train]),('passenger',raw.PassengerId.iloc[query])]:
                np.testing.assert_array_equal(z[key],v)
            score[:,:,query] = z['p']
        print(int(k),flush=True)
    assert np.isfinite(score).all()
    report = {}
    for tag in ['full','bookingmix','historical_to_new']:
        valid,parent = views[tag]
        wrong = parent!=y
        old = (votes==parent[:,None]).all(1)
        new = ~paid&~old&(votes[:,:-2]==parent[:,None]).all(1)
        scopes = dict(all=valid,new_consensus=valid&new,new_G=valid&new&(x.Deck=='G').to_numpy(),
                      old_consensus=valid&old,remaining=valid&~(old|new))
        r = {}
        avg = score.mean(axis=0)
        oppose = (avg>=.5)!=parent
        for name, mask in scopes.items():
            rr = {}
            for j,n in enumerate(NAMES):
                rr[n] = dict(**ticker(mask,wrong,oppose[j]),log_loss=float(log_loss(y[mask],avg[j,mask],labels=[False,True])))
            for j,n in [(1,'real'),(2,'shuffled')]:
                rr[n+'_new'] = ticker(mask,wrong,oppose[j]&~oppose[0])
                rr[n+'_withdrawn'] = ticker(mask,wrong,~oppose[j]&oppose[0])
                # Require all training rotations to support the same neighbor-driven edit.
                tipped = (((score[:,j]>=.5)!=parent)&((score[:,0]>=.5)==parent)).all(0)
                rr[n+'_three_rotations'] = ticker(mask,wrong,tipped)
            r[name] = rr
        report[tag] = r
        print(tag,r['all']['real_new'],r['all']['shuffled_new'],r['new_G']['real_neighbors'],flush=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
