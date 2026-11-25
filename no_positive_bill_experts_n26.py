"""Test specialists for passengers with no observed positive charge, using unlabeled companions."""
import json
import numpy as np
import pandas as pd
from booking_stream_challenge_n26 import stream
from full_roster_booking_validation_n23 import ROOT
from held_expert_portfolios_n26 import DECK, panel_predictions, replay, sources, splits
from outside_panel_coverage_n26 import projection
from price_implied_deck_n26 import FEES
from solo_selection_likelihood_n26 import fit_model
from solo_archive_recovery_n26 import ticker

OUT = ROOT/'artifacts/nov26_no_positive_experts'


def design(raw, group):
    x = stream(raw)[0]
    paid = raw[FEES].gt(0).any(axis=1)
    siblings = x.party_size - 1
    peer = pd.DataFrame(index=x.index)
    fields = pd.DataFrame({**{service:np.log1p(raw[service]) for service in FEES},
                           'paid':paid, 'sleep':raw.CryoSleep.map({True:1.,False:0.}), 'age':raw.Age})
    for name, values in fields.items():
        known = values.notna().astype(int)
        total = values.fillna(0.).astype(float)
        peer[name] = (total.groupby(group).transform('sum')-total) / (known.groupby(group).transform('sum')-known).replace(0,np.nan)
    x['unknown_fees'] = raw[FEES].isna().sum(axis=1)
    # Transfer entire companion profiles between different orders of the same size and home planet.
    moved = peer.copy()
    moved.iloc[:] = np.nan
    rng = np.random.default_rng(9961)
    for _, ids in x.groupby(['party_size','HomePlanet'],sort=True).indices.items():
        orders = rng.permutation(np.unique(group[ids]))
        if len(orders)<2:
            continue
        donors = dict(zip(orders,np.roll(orders,1),strict=True))
        for gid, positions in pd.Series(ids,index=group[ids]).groupby(level=0):
            recipient = positions.to_numpy()
            donor = ids[group[ids]==donors[gid]]
            moved.iloc[recipient] = peer.iloc[rng.choice(donor,len(recipient),replace=True)].to_numpy()
    assert peer[siblings==0].isna().all().all() and moved[siblings==0].isna().all().all()
    return x,peer.add_prefix('peer_'),moved.add_prefix('peer_'),paid.to_numpy()


def run():
    raw,y,g,fold,views,votes = projection()
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns.tolist())
    people = pd.concat([raw,test],ignore_index=True)
    group = people.PassengerId.str.split('_').str[0].to_numpy()
    reg,peer,sham,paid = design(people,group)
    dry = ~paid[:len(y)]
    views = {k:v for k,v in views.items() if k in ('bookingmix','historical_to_new')}
    v = sources()[-1]
    full = v['full']
    panels = panel_predictions(full,'full')
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible = z['eligible']
    mask = np.zeros(len(y),bool)
    current = mask.copy()
    for key,_,dst,_,q in splits(v,g):
        if key.startswith('full_'):
            assert not mask[q].any()
            mask[q] = True
            current[q] = replay(key,q,dst,panels,eligible)
    views['full'] = mask,current
    models = {'all_registration':reg, 'dry_registration':reg,
              'dry_companions':pd.concat([reg,peer],axis=1), 'dry_shuffled_companions':pd.concat([reg,sham],axis=1)}
    forecasts = {n:np.full(len(y),np.nan) for n in models}
    OUT.mkdir(exist_ok=True)
    for k in np.unique(fold):
        t,q = np.flatnonzero(fold!=k),np.flatnonzero((fold==k)&dry)
        path = OUT/f'{k}.npz'
        assert not np.isin(g[t],g[q]).any()
        if not path.exists():
            values = {}
            for name,x in models.items():
                use=t if name=='all_registration' else t[dry[t]]
                values[name]=fit_model(x.iloc[use],y[use]).predict_proba(x.iloc[q])[:,1]
            with path.with_suffix('.tmp').open('wb') as h:
                np.savez_compressed(h,t=t,q=q,y=y[t],**values)
            path.with_suffix('.tmp').replace(path)
        with np.load(path) as z:
            for name,value in [('t',t),('q',q),('y',y[t])]:
                np.testing.assert_array_equal(z[name],value)
            for name,p in forecasts.items():
                p[q] = z[name]
        print(k,flush=True)
    report = {}
    for tag,(valid,parent) in views.items():
        wrong = parent!=y
        # The last two columns are registration and sequence classifiers fitted only on payers.
        from held_expert_portfolios_n26 import NEW
        assert NEW[-2:] == ['register_paid','sequence_paid']
        old = (votes==parent[:,None]).all(1)
        without_paid = (votes[:,:-2]==parent[:,None]).all(1)
        scopes = {'all_dry':valid&dry,'new_unanimous':valid&dry&without_paid&~old,
                  'other_dry':valid&dry&~without_paid,'dry_with_companions':valid&dry&(reg.party_size.to_numpy()[:len(y)]>1)}
        for n,p in forecasts.items():
            assert np.isfinite(p[dry]).all()
        moves = {n:dry&((p>=.5)!=parent) for n,p in forecasts.items()}
        moves['both_dry_models'] = moves['dry_companions'] & moves['dry_registration']
        moves['only_real_peers'] = moves['dry_companions'] & ~moves['dry_registration'] & ~moves['dry_shuffled_companions']
        r = {s:{n:ticker(m,wrong,action) for n,action in moves.items()} for s,m in scopes.items()}
        print(tag,r['all_dry'],r['new_unanimous']['dry_companions'],flush=True)
        report[tag] = r
    (ROOT/'reports/nov23_matched_learning_curve/no_positive_bill_experts.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
