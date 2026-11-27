"""Compare direction-specific foundation-model corrections on disjoint calibration and evaluation orders."""
import json
import numpy as np
import pandas as pd
from repeated_archive_misses_n26 import ART,cohort,count
from full_roster_booking_validation_n23 import ROOT,data
from booking_mix_confidence_transfer_n26 import partition
from six_way_against_scored_nine_n26 import test_decisions
from export_joint_fare_foundations_n26 import NAME as NEW, BASE
from threadpoolctl import threadpool_limits

OUT=ART/'nov27_directional_foundation'
REPORT=ROOT/'reports/nov23_matched_learning_curve/directional_foundation_n27.json'


def calibrate(risk,base,truth,rows,shrink):
    # Each side gets its own cut, including a no-edit option. Penalize variance on short prefixes.
    cut=np.full(2,np.inf)
    for side in (0,1):
        ix=rows[(base[rows]==side)&(risk[rows]>.5)]
        ix=ix[np.argsort(-risk[ix],kind='stable')]
        if not len(ix):continue
        delta=np.cumsum(np.where(base[ix]!=truth[ix],1,-1))
        valid=np.r_[risk[ix][:-1]!=risk[ix][1:],True]
        val=np.where(valid,delta-shrink*np.sqrt(np.arange(1,len(ix)+1)),-np.inf)
        if val.max()>0:cut[side]=risk[ix[val.argmax()]]
    return cut


def run():
    raw,y,g,f=data()
    n=len(y)
    baseline,blind,known,fresh=cohort(raw,y,f)
    ids=np.r_[raw.PassengerId.to_numpy(str),pd.read_csv(ROOT/'data/raw/test.csv').PassengerId.to_numpy(str)]
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        first=np.r_[z['champion111'],test_decisions(BASE,pd.Series(ids[n:]))[0]].astype(bool)
    parent=np.r_[baseline,test_decisions(NEW,pd.Series(ids[n:]))[0]]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_array_equal(z['anchor'],parent)
        names=z['names']
        assert names[-3:].tolist()==['Mitra','TabICL fare','Nori fare']
        cols=z['scores'][:,-3:].copy()
    with np.load(ART/'nov26_surprise_sampled_foundation/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['y'],y)
        augmented=z['full']
    opinions=dict(Mitra=cols[:,0],TabICL=cols[:,1],Nori=cols[:,2],TabICL_augmented=augmented,
                  median=np.median(cols,axis=1),consensus=np.where(first,cols.max(1),cols.min(1)))
    split=partition(g,'directional_foundation',4)
    report={};test_actions={}
    for k,p in opinions.items():
        opposite=np.where(first,1-p,p)
        actions={}
        for s in [0.,1.]:
            edit=np.zeros(len(ids),bool)
            votes=[]
            for j in np.unique(f):
                for m in [0,1,2,3,None]:
                    tr=np.flatnonzero(known&(f==j)&((split!=m) if m is not None else True))
                    q=np.flatnonzero((f==j)&(split==m)) if m is not None else np.arange(n,len(ids))
                    cut=calibrate(opposite,first,y,tr,s)
                    v=(opposite[q]>=cut[first[q].astype(int)])
                    if m is None:votes.append(v)
                    else:edit[q]=v
            edit[n:]=np.mean(votes,axis=0)>.5
            actions[f'held_{s:g}']=edit
        # Uniform opinion and the subset where three foundations disagree unanimously with current release.
        actions['ordinary']=opposite>.5
        actions['majority_confirmed']=actions['held_1'] & (np.sum((cols>=.5)!=first[:,None],axis=1)>=2)
        for tag,a in actions.items():
            key=k+'_'+tag
            pred=first^a
            test_actions[key]=pred[n:]
            d=(pred[:n]==y).astype(int)-(first[:n]==y)
            r=dict(test_changes=int(a[n:].sum()),folds=[int(d[known&(f==j)].sum()) for j in range(8)],scopes={})
            for scope,u in [('all',known),('fresh',known&fresh),('blind',blind),('positive_base',known&first[:n]),('negative_base',known&~first[:n])]:
                r['scopes'][scope]=count(a[:n],u,first[:n]!=y)
            report[key]=r
            rr=r['scopes']['all'];fr=r['scopes']['fresh']
            print(key,rr['fixed']-rr['harmed'],fr['fixed']-fr['harmed'],r['test_changes'])
    median=opinions['median']
    opposite=np.where(first,1-median,median)
    ttest=np.arange(n,len(ids))
    cuts=np.stack([calibrate(opposite,first,y,np.flatnonzero(known&(f==j)),1.) for j in range(8)])
    np.testing.assert_array_equal((np.mean(opposite[ttest,None]>=cuts[:,first[ttest].astype(int)].T,axis=1)>.5)^first[ttest],test_actions['median_held_1'])
    extra={}
    for protocol in ('contiguous','multiple_random'):
        parts=[np.zeros(n,int)] if protocol=='contiguous' else [partition(g,'confirm_foundation_'+str(seed),4) for seed in range(8)]
        if protocol=='contiguous':
            for j in range(8):
                for m,book in enumerate(np.array_split(np.unique(g[f==j]),4)):
                    parts[0][(f==j)&np.isin(g,book)]=m
        votes=np.zeros(n,int)
        for part in parts:
            for j in range(8):
                for m in range(4):
                    t=np.flatnonzero((f==j)&(part!=m)&known)
                    q=np.flatnonzero((f==j)&(part==m))
                    assert not np.isin(g[t],g[q]).any()
                    cut=calibrate(opposite,first,y,t,1.)
                    votes[q]+=(opposite[q]>=cut[first[q].astype(int)])
        action=votes>len(parts)/2
        d=action*np.where(first[:n]!=y,1,-1)
        extra[protocol]=dict(folds=[int(d[known&(f==j)].sum()) for j in range(8)],
             scopes={k:count(action,m,first[:n]!=y) for k,m in [('all',known),('fresh',known&fresh),('negative',known&~first[:n]),('positive',known&first[:n])]})
    report['median_confirmation']=extra
    print(extra)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'test.npz',passenger=ids[n:],**test_actions)
    np.savez_compressed(OUT/'thresholds.npz',cuts=cuts,passenger=ids[n:],base=first[n:],scores=cols[n:])
    REPORT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(1):run()
