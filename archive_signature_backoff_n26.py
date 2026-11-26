"""Use recurring archive vote patterns to estimate outcomes on unseen bookings, with nested backoff."""
import json

import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from booking_mix_confidence_transfer_n26 import partition
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT, data
from kaggle_workflow import sha256 as file_sha
from repeated_archive_misses_n26 import ART, OUT as BANK, cohort, count

OUT = ART/'nov26_archive_signature_backoff'
REPORT = ROOT/'reports/nov23_matched_learning_curve/archive_signature_backoff.json'


def representations(scores, groups):
    # All configurations from one project experiment have one summary vote, including tied/abstaining votes.
    family = np.unique(groups)
    signed = np.sign(scores - .5)
    return np.stack([np.sign(signed[:, groups==g].sum(1)).astype(np.int8)+1 for g in family], axis=1)


def oracle(bits, base, y, scope):
    totals = {}
    for ix in np.flatnonzero(scope):
        key = (bool(base[ix]), bits[ix].tobytes())
        hist = totals.setdefault(key,[0,0])
        hist[int(base[ix]!=y[ix])] += 1
    return dict(patterns=len(totals), singles=sum(sum(v)==1 for v in totals.values()),
                errors_in_indistinguishable_patterns=sum(v[1] for v in totals.values() if min(v)>0),
                hindsight_net=sum(max(0, b-a) for a,b in totals.values()))


def paths(bits):
    rng = np.random.default_rng(55047)
    # The order and length of each prefix use no labels, confidence thresholds or outcome selection.
    walks = [rng.permutation(bits.shape[1])[:10] for _ in range(48)]
    return [np.cumsum(bits[:,cols].astype(np.int32)*(3**np.arange(len(cols))),axis=1) for cols in walks]


def estimate(walks,t,q,y,prior):
    answer = np.zeros(len(q))
    for walk in walks:
        score = np.full(len(q),(1+y[t].sum())/(len(t)+2))
        for k in range(walk.shape[1]):
            size = 3**(k+1)
            code = walk[:,k]
            mass = np.bincount(code[t], minlength=size)[code[q]]
            hits = np.bincount(code[t],weights=y[t],minlength=size)[code[q]]
            if prior:
                score = (hits+8*score)/(mass+8)
            else:
                score = np.where(mass>=8, (hits+1)/(mass+2), score)
        answer += score
    return answer/len(walks)


def run():
    raw,y,g,f=data()
    base, blind, known, fresh=cohort(raw,y,f)
    n=len(y)
    with np.load(BANK/'aligned.npz') as z:
        for key,expected in [('passenger',raw.PassengerId),('fold',f),('y',y)]:
            np.testing.assert_array_equal(z[key],expected)
        archival, family = z['full'], z['families']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        anchors = z['anchor']
        passenger = z['passenger']
        prob = z['scores'][:,-3:].mean(1)
    np.testing.assert_array_equal(anchors[:n],base)
    _,_,_,_,names,full,*_=panels_and_reference()
    archive=ART/'proposal_n26_dispute_band/frozen_votes.npz'
    manifest=json.loads((ROOT/'reports/proposal_n26_dispute_band/manifest.json').read_text())
    assert file_sha(archive)==manifest['model_sha256']
    with np.load(archive) as z:
        np.testing.assert_array_equal(z['names'],names)
        np.testing.assert_array_equal(z['passenger'],passenger[n:])
        deploy=np.vstack([full,z['p']])
    bank={'wide':(archival,family), 'deployable':(deploy,np.array([s.split('/')[0] for s in names]))}
    result={}
    cache={}
    inner=partition(g,'vote_signature_validation',4)
    for key,(scores,families) in bank.items():
        bits=representations(scores,families)
        baseline_strength=np.where(anchors[:len(scores)],1.,-1.)*abs(logit(prob[:len(scores)].clip(.01,.99)))
        train=np.arange(n)
        submit=np.arange(n,len(scores))
        outcome={'oracle':{k:oracle(bits,base,y,mask) for k,mask in [('known',known),('blind',blind)]}}
        individual = np.sign(scores-.5).astype(np.int8)+1
        outcome['oracle_by_configuration'] = {scope:oracle(individual,base,y,mask) for scope,mask in [('known',known),('blind',blind)]}
        opinions={}
        for method,features in [('identity',bits),('counts',np.sort(bits,axis=1))]:
            walks=paths(features)
            for smooth in (False,True):
                p=np.full(len(scores),np.nan)
                tests=[]
                for k in np.unique(f):
                    for s in [0,1,2,3,None]:
                        t=train[(f==k)&((inner!=s) if s is not None else True)]
                        q=train[(f==k)&(inner==s)] if s is not None else submit
                        if not len(q):
                            continue
                        if s is not None:
                            assert not np.isin(g[t],g[q]).any()
                        z=estimate(walks,t,q,y,smooth)
                        if s is None: tests.append(z)
                        else:p[q]=z
                if len(submit):p[submit]=np.mean(tests,axis=0)
                assert np.isfinite(p).all()
                tag=method+('_shrink' if smooth else '_backoff')
                opinions[tag]=p
                cache[key+'_'+tag]=p
        flags={k:(v>=.5)!=anchors[:len(v)] for k,v in opinions.items()}
        flags['new_identity']=flags['identity_shrink']&flags['identity_backoff']&~flags['counts_shrink']&~flags['counts_backoff']
        conditional=expit(baseline_strength+logit(opinions['identity_shrink'])-logit(opinions['counts_shrink']))
        flags['conditional_identity']=(conditional>=.5)!=anchors[:len(scores)]
        for tag,v in flags.items():
            delta=v[:n]*np.where(base!=y,1,-1)
            outcome[tag]=dict(test_edits=int(v[n:].sum()) if len(submit) else None,
                scopes={a:count(v[:n],m,base!=y) for a,m in [('known',known),('blind',blind),('fresh',fresh&known)]},
                net_by_fold=[int(delta[known&(f==k)].sum()) for k in np.unique(f)])
        result[key]=outcome
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'predictions.npz',passenger=passenger,fold=f,y=y,**cache)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    for scope,items in result.items():
        print(scope,items['oracle']['blind'],[(k,v['scopes']['known']) for k,v in items.items() if 'scopes' in v])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
