"""Transfer the champion's unchanged six-voter rule to missing and zero-positive receipts."""
import argparse
import json
import numpy as np
import pandas as pd

from arbitration_against_scored_nine_n26 import champions
from booking_stream_challenge_n26 import stream
from export_six_way_dispute_n26 import combine
from full_roster_booking_validation_n23 import ROOT,data
from held_expert_portfolios_n26 import DECK,SIX,panel_predictions,sources
from kaggle_workflow import sha256
from rebase_fare_six_n26 import NAME as BASE,predict
from repeated_archive_misses_n26 import ART,cohort,count
from price_implied_deck_n26 import FEES
from solo_selection_likelihood_n26 import fit_model

OUT=ART/'nov27_extend_six_payment_states'
REPORT=ROOT/'reports/nov23_matched_learning_curve/extend_six_payment_states.json'


def specialists(frame,y,groups,fold,paid):
    reg,seq,_=stream(frame)
    extended=pd.concat([reg,seq],axis=1)
    n=len(y)
    outputs={key:np.full((len(frame),2),np.nan) for key in ('specialist','matched_prior')}
    for k in range(9):
        t=np.flatnonzero(fold!=k) if k<8 else np.arange(n)
        q=np.flatnonzero(fold==k) if k<8 else np.arange(n,len(frame))
        assert not np.isin(groups[t],groups[q]).any()
        path=OUT/f'fold{k}.npz'
        zero=t[~paid[t]]
        rng=np.random.default_rng(22409+k)
        control=np.concatenate([rng.choice(t[y[t]==v],np.sum(y[zero]==v),replace=False) for v in (0,1)])
        if not path.exists():
            scores={}
            for key,ids in [('specialist',zero),('matched_prior',control)]:
                scores[key]=np.column_stack([fit_model(x.iloc[ids],y[ids]).predict_proba(x.iloc[q])[:,1] for x in (reg,extended)])
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,passenger=frame.PassengerId.iloc[q].to_numpy(str),t=t,q=q,labels=y[t],
                                zero=zero,control=control,**scores)
            tmp.replace(path)
        with np.load(path) as z:
            for a,b in [('passenger',frame.PassengerId.iloc[q]),('t',t),('q',q),('labels',y[t]),('zero',zero),('control',control)]:
                np.testing.assert_array_equal(z[a],b)
            for key,p in outputs.items():p[q]=z[key]
    return outputs


def run(train):
    raw,y,g,f=data()
    n=len(y)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    ids=frame.PassengerId.to_numpy(str)
    groups=frame.PassengerId.str[:4].to_numpy()
    size=frame.groupby(groups).PassengerId.transform('size').to_numpy()
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    complete=frame[FEES].notna().all(axis=1).to_numpy()
    scopes=dict(positive_incomplete=paid&~complete,zero_complete=~paid&complete,zero_incomplete=~paid&~complete)
    outside=~(paid&complete)
    base161,blind,known,fresh=cohort(raw,y,f)
    old,verified,oldhist,hi=champions()
    np.testing.assert_array_equal(known,verified)
    v=sources()[-1]
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['eligible'],~outside)
    hist={}
    for tag,d in v.items():
        ix=d['ids']
        p=panel_predictions(d,tag)
        six=np.column_stack([p[j] for j in SIX])
        fam=d['votes'][:,:18]
        base=predict((old if tag=='full' else oldhist)[ix],p['neural_deck_raw'],p['neural_deck_fine'],
                     six,fam,~outside[ix])['joint']>=.5
        if tag=='full':
            np.testing.assert_array_equal(ix,np.arange(n))
            np.testing.assert_array_equal(base[outside[:n]&known],base161[outside[:n]&known])
            full=(base,six,fam)
        else:
            hist_scope=outside[ix]
            act=combine(base,six,fam,hist_scope)>=.5
            hist={key:count(act!=base,mask[ix],base!=y[ix]) for key,mask in scopes.items()}
    path=ART/BASE/'scores.npz'
    assert sha256(path)==json.loads((ROOT/'reports'/BASE/'manifest.json').read_text())['model_sha256']
    with np.load(path) as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:])
        six=np.vstack([full[1],z['challengers']])
        fam=np.vstack([full[2],z['family_votes']])
        test_base=predict(z['anchor'],z['raw'],z['fine'],z['challengers'],z['family_votes'],z['eligible'])['joint']>=.5
    incumbent=pd.read_csv(ART/BASE/'submission.csv')
    np.testing.assert_array_equal(incumbent.PassengerId,ids[n:])
    np.testing.assert_array_equal(incumbent.Transported,test_base)
    base=np.r_[full[0],test_base]
    assert np.isfinite(six).all()
    OUT.mkdir(exist_ok=True)
    flags={'frozen_six':(combine(base,six,fam,outside)>=.5)!=base}
    flags['six_without_family']=outside&((six>=.5)!=base[:,None]).all(1)
    flags['family_only']=outside&((fam!=base[:,None]).sum(1)>=10)
    # Two paid-only registry experts in the frozen six are extrapolating for zero-positive passengers.
    for key,rep in (specialists(frame,y,groups,f,paid) if train else {}).items():
        panel=six.copy()
        panel[~paid,1:3]=rep[~paid]
        flags[key]=(combine(base,panel,fam,outside)>=.5)!=base
        flags[key+'_without_family']=outside&((panel>=.5)!=base[:,None]).all(1)
    scopes.update(outside=outside,single=outside&(size==1),pair=outside&(size==2),larger=outside&(size>=3))
    report=dict(historical_frozen=hist,rules={})
    if train:
        flags['specialist_only']=flags['specialist']&~flags['frozen_six']&~flags['matched_prior']
        flags['registry_transport']=flags['specialist_without_family'] & ~flags['family_only'] & ~flags['matched_prior_without_family'] & ~paid
        flags['replicated_zero']=flags['specialist']&flags['frozen_six']&~paid
    for key,flag in flags.items():
        delta=flag[:n]*np.where(base[:n]!=y,1,-1)
        report['rules'][key]={}
        for s,m in scopes.items():
            seg=m[:n]&known
            report['rules'][key][s]=dict(rows=int(seg.sum()),test_changes=int((flag[n:]&m[n:]).sum()),
                all=count(flag[:n],seg,base[:n]!=y),fresh=count(flag[:n],seg&fresh,base[:n]!=y),
                blind=count(flag[:n],seg&blind,base[:n]!=y),folds=[int(delta[seg&(f==i)].sum()) for i in range(8)])
    np.savez_compressed(OUT/'actions.npz',passenger=ids,y=y,fold=f,base=base,six=six,family=fam,**flags)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print({k:{s:(d['all'],d['test_changes']) for s,d in v.items() if s in ('outside','positive_incomplete','zero_complete','zero_incomplete')} for k,v in report['rules'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-only',action='store_true')
    run(not p.parse_args().frozen_only)
