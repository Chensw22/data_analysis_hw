"""Learn specifically on passengers without positive bills whose companions have complete paid bills."""
import argparse
import json

import numpy as np
import pandas as pd

from exaone_validation_data import compact_features
from full_roster_booking_validation_n23 import ROOT,data
from nori_fare_party_n26 import DECKS,FEES,features

OUT=ROOT/'artifacts/nov26_unpaid_companion_fare'
REPORT=ROOT/'reports/nov23_matched_learning_curve/unpaid_companion_fare.json'


def design(raw,g):
    x=compact_features(raw)
    fees=raw[FEES]
    paid=fees.notna().all(axis=1)&fees.gt(0).any(axis=1)
    x['party_size']=raw.PassengerId.groupby(g).transform('size')
    x['recorded_payers']=paid.groupby(g).transform('sum')
    payer_cost=fees.sum(axis=1,min_count=5).where(paid)
    x['companion_mean_total']=np.log1p(payer_cost.groupby(g).transform('mean'))
    for name in FEES:
        x['companion_log_'+name]=np.log1p(fees[name].where(paid).groupby(g).transform('mean'))
    use=(~fees.gt(0).any(axis=1)&(x.recorded_payers>0)).to_numpy()
    return x,use


def substitute(x,g,t,q,seed):
    # Preserve approximate fare size and party composition while removing the actual relationship.
    rng=np.random.default_rng(seed)
    order=rng.permutation(t)
    planet,deck=x.HomePlanet.to_numpy(),x.Deck.to_numpy()
    size=x.party_size.to_numpy()
    payers=x.recorded_payers.to_numpy()
    costs=x.companion_mean_total.to_numpy()
    out=np.zeros(len(q),int)
    for j,i in enumerate(q):
        penalty=3*(planet[order]!=planet[i])+3*(deck[order]!=deck[i])
        penalty=penalty+.7*abs(size[order]-size[i])+.7*abs(payers[order]-payers[i])+abs(costs[order]-costs[i])
        penalty[g[order]==g[i]]=np.inf
        out[j]=order[np.argmin(penalty)]
    assert np.isin(out,t).all() and (g[out]!=g[q]).all()
    return out


def train(arch):
    if arch=='tabicl':
        import hashlib
        import torch
        from tabicl_routed_context_n23 import DIGEST,WEIGHT,infer
        torch.set_num_threads(2)
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==DIGEST
    else:
        from solo_selection_likelihood_n26 import fit_model
    raw,y,g,f=data()
    x,use=design(raw,g)
    OUT.mkdir(exist_ok=True)
    for k in np.unique(f):
        path=OUT/f'{arch}{k}.npz'
        if path.exists():
            continue
        outer_t,outer_q=np.flatnonzero(f!=k),np.flatnonzero(f==k)
        _,peer,counts,_=features(raw,g,outer_t,outer_q)
        t,q=outer_t[use[outer_t]],outer_q[use[outer_q]]
        assert not np.isin(g[t],g[q]).any() and (counts[use]>0).all()
        ids=np.r_[t,q]
        alt=substitute(x,g,t,ids,4471+int(k))
        frames={'basic':x}
        for name,values in [('linked',peer[ids,1:]),('matched_other',peer[alt,1:])]:
            frame=x.copy()
            for j,deck in enumerate(DECKS):
                frame['peer_fare_'+deck]=np.nan
                frame.loc[ids,'peer_fare_'+deck]=values[:,j]
            frames[name]=frame
        output={}
        for name,frame in frames.items():
            if arch=='tabicl':
                output[arch+'_'+name]=infer(frame,y,t,q,6551+int(k))
            else:
                output[arch+'_'+name]=fit_model(frame.iloc[t],y[t]).predict_proba(frame.iloc[q])[:,1]
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,t=t,q=q,train_y=y[t],passenger=raw.PassengerId.iloc[q].to_numpy(str),donor=alt,**output)
        tmp.replace(path)
        print('specialist fold',int(k),len(q),flush=True)


def report():
    from arbitration_against_scored_nine_n26 import champions
    from exhaustive_family_quorums_n26 import inputs, summary
    from held_expert_portfolios_n26 import DECK,SIX,panel_predictions,sources
    from rebase_fare_six_n26 import predict
    raw,y,g,f=data()
    _,use=design(raw,g)
    v=sources()[-1]['full']
    p=panel_predictions(v,'full')
    a,known,_,hi=champions()
    with np.load(DECK/'deck_estimates.npz') as z:
        eligible=z['eligible'][:len(y)]
    parent=predict(a,p['neural_deck_raw'],p['neural_deck_fine'],np.column_stack([p[n] for n in SIX]),v['votes'][:,:18],eligible)['joint']>=.5
    np.testing.assert_array_equal(parent[use],a[use])
    _,cur,_=inputs()
    predictions={}
    for k in np.unique(f):
        t,q=np.flatnonzero((f!=k)&use),np.flatnonzero((f==k)&use)
        for arch in ['tree','tabicl']:
            with np.load(OUT/f'{arch}{k}.npz') as z:
                for key,val in [('q',q),('t',t),('train_y',y[t]),('passenger',raw.PassengerId.iloc[q].to_numpy(str))]:
                    np.testing.assert_array_equal(z[key],val)
                assert np.isin(z['donor'],t).all() and (g[z['donor']]!=g[np.r_[t,q]]).all()
                for key in [n for n in z.files if n.startswith(('tree_','tabicl_'))]:
                    probabilities=z[key]
                    assert np.isfinite(probabilities).all()
                    predictions.setdefault(key,np.full(len(y),np.nan))[q]=probabilities
    scopes={'all_specialists':use&known,'previously_unanimous':use&cur['common'],'outside_history':use&known&~np.isin(g,g[hi])}
    acts={}
    for arch in ['tabicl','tree']:
        base,linked,control=[predictions[arch+'_'+n]>=.5 for n in ['basic','linked','matched_other']]
        acts[arch]=use&(linked!=parent)
        acts[arch+'_specific']=acts[arch]&(base==parent)&(control==parent)
    acts['both']=acts['tabicl']&acts['tree']
    acts['both_specific']=acts['tabicl_specific']&acts['tree_specific']
    out=dict(rows=int(use.sum()),score={k:float(((v[use]>=.5)==y[use]).mean()) for k,v in predictions.items()},
             against_clean_champion={k:{s:summary(act,parent!=y,m,f) for s,m in scopes.items()} for k,act in acts.items()})
    groups,idx=np.unique(g[use],return_inverse=True)
    mass=np.bincount(idx)
    rng=np.random.default_rng(452)
    out['linked_vs_matched']={}
    for arch in ['tree','tabicl']:
        linked=predictions[arch+'_linked'][use]
        matched=predictions[arch+'_matched_other'][use]
        truth=y[use]
        diff=((linked>=.5)==truth).astype(int)-((matched>=.5)==truth)
        per_order=np.bincount(idx,weights=diff)
        draws=rng.integers(len(groups),size=(2000,len(groups)))
        effect=100*per_order[draws].sum(1)/mass[draws].sum(1)
        out['linked_vs_matched'][arch]=dict(net=int(diff.sum()),accuracy_gain_pp=float(100*diff.mean()),
            booking_bootstrap_95_pp=np.quantile(effect,[.025,.975]).tolist(),
            brier_change=float(np.mean((linked-truth)**2-(matched-truth)**2)),
            fold_net=[int(diff[f[use]==k].sum()) for k in np.unique(f)])
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    print(out['score'],[(k,v['all_specialists']['fixed'],v['all_specialists']['harmed']) for k,v in out['against_clean_champion'].items()])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['tabicl','tree','report'])
    action=parser.parse_args().action
    if action!='report':
        train(action)
    else:
        report()
