"""Route paired fare-model changes using the direction and reproducibility of their evidence."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from held_expert_portfolios_n26 import sources,panel_predictions
from booking_mix_confidence_transfer_n26 import partition
from six_way_against_scored_nine_n26 import test_decisions
from export_directional_foundation_n27 import NAME as CHAMP
from export_joint_fare_foundations_n26 import NAME as JOINT,BASE
from export_consensus_review_n26 import dig,write
from kaggle_workflow import validate_submission

REPORT=ROOT/'reports/nov23_matched_learning_curve/fifty_fare_adjudication_n27.json'
STORE=ART/'nov27_fifty_fare_adjudication'
NAME='proposal_n27_fifty_fare_adjudication'


def run(chosen=None):
    raw,y,g,f=data()
    _,blind,known,fresh=cohort(raw,y,f)
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    n=len(y)
    ids=np.r_[raw.PassengerId.to_numpy(str),test.PassengerId.to_numpy(str)]
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        np.testing.assert_array_equal(z['y'],y)
        parent=z['parent']
    np.testing.assert_array_equal(parent[n:],test_decisions(CHAMP,test.PassengerId)[0])
    v=sources()[-1]['full']
    pp=panel_predictions(v,'full')
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        first=z['champion111'].astype(bool)
        raw_score=np.column_stack([pp['neural_deck_raw'],z['raw']])
        fine=np.column_stack([pp['neural_deck_fine'],z['own']])
        paid=z['eligible'].astype(bool)
    with np.load(ART/JOINT/'inputs.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:])
        raw_score=np.vstack([raw_score,z['raw']])
        fine=np.vstack([fine,z['fine']])
        first=np.r_[first,z['parent']>=.5]
        paid=np.r_[paid,z['eligible']]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        joint=z['anchor']
        opinion=z['scores']
    np.testing.assert_array_equal(joint[n:],test_decisions(JOINT,test.PassengerId)[0])
    assert np.sum(first[n:]!=joint[n:])==50 and np.sum(parent[n:]!=first[n:])==16
    scope=(first!=joint)&(parent==first)
    sign=np.where(first,-1.,1.)[:,None]
    old=(raw_score-.5)*sign
    after=(fine-.5)*sign
    gain=after-old
    family=((opinion[:,:18]>=.5)!=first[:,None]).sum(1)
    native=(opinion[:,-3]-.5)*sign[:,0]
    dual=(after>0).all(1)
    include=paid & dual
    assert np.all(include[scope])
    regs=dict(increasing=(gain>0).all(1),crossed=(old<=0).any(1),native=native>0,
              family_half=family>=9,positive=~first,negative=first,
              nori_strengthens=gain[:,1]>0,tab_strengthens=gain[:,0]>0,
              nori_crosses=old[:,1]<=0,tab_crosses=old[:,0]<=0)
    acts={'fixed_'+key:scope&v for key,v in regs.items()}
    kept={}
    for mode in ['random','ordered']:
        bins=partition(g,'paired_50_calibration',4) if mode=='random' else np.zeros(n,int)
        controls={'level':np.column_stack([first,after,after.min(1)]),
                  'gain':np.column_stack([first,after,after.min(1),gain,old.min(1),native,family/18])}
        for kind,x in controls.items():
            score=np.full(len(ids),np.nan);tests=[]
            for k in range(8):
                if mode=='ordered':
                    for h,grp in enumerate(np.array_split(np.unique(g[f==k]),4)):
                        bins[(f==k)&np.isin(g,grp)]=h
                for j in [0,1,2,3,None]:
                    t=np.flatnonzero((f==k)&known&paid[:n]&((bins!=j) if j is not None else True))
                    q=np.arange(n,len(ids)) if j is None else np.flatnonzero((f==k)&(bins==j))
                    # Calibrate from every paid held-out passenger in training. Emphasize dual dissent without using test labels.
                    weights=np.where(dual[t],3.,1.)
                    mean=x[t].mean(0);sd=x[t].std(0).clip(.08)
                    reg=LogisticRegression(C=.07,max_iter=300).fit((x[t]-mean)/sd,y[t]!=first[t],sample_weight=weights)
                    p=reg.predict_proba((x[q]-mean)/sd)[:,1]
                    if j is None:tests.append(p)
                    else:score[q]=p
            score[n:]=np.mean(tests,axis=0)
            kept[kind+'_'+mode]=score
            for key,m in [('fifty',scope),('extension',paid & (first==parent)&(after.max(1)>0))]:
                acts[kind+'_'+key+'_'+mode]=m&(score>.5)
        acts['gain_only_fifty_'+mode]=scope&(kept['gain_'+mode]>.5)&(kept['level_'+mode]<=.5)
        acts['corroborated_fifty_'+mode]=scope&(kept['gain_'+mode]>.5)&(kept['level_'+mode]>.5)
    for key,action in list(acts.items()):
        if key.startswith('fixed_'):
            continue
        acts[key+'_up']=action&~first
        acts[key+'_down']=action&first
    report={}
    for key,edit in acts.items():
        d=np.where(parent[:n]!=y,1,-1)*edit[:n]
        report[key]=dict(scopes={name:count(edit[:n],m,parent[:n]!=y) for name,m in [('all',known),('fresh',known&fresh),('common',blind)]},
            by_fold=[int(d[known&(f==k)].sum()) for k in range(8)],test=int(edit[n:].sum()),in_50=int((edit[n:]&(joint[n:]!=first[n:])).sum()))
    STORE.mkdir(exist_ok=True)
    np.savez_compressed(STORE/'scores.npz',ids=ids,y=y,fold=f,parent=parent,first=first,joint=joint,raw=raw_score,fine=fine,**acts)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for key,r in report.items(): print(key,r['scopes']['all'],r['scopes']['fresh'],r['by_fold'],r['test'])
    if chosen:
        dest=ART/NAME
        dest.mkdir(exist_ok=True)
        p=pd.read_csv(ART/CHAMP/'test_probabilities.csv',float_precision='round_trip')
        np.testing.assert_array_equal(p.PassengerId,test.PassengerId)
        base=p.probability.to_numpy()
        change=acts[chosen][n:]
        assert chosen=='level_fifty_random_up'
        model=dest/'inputs.npz'
        pred=kept['level_random'][n:]
        np.testing.assert_array_equal(change,acts['level_fifty_ordered_up'][n:])
        np.savez_compressed(model,passenger=test.PassengerId.to_numpy(str),parent=base,first=first[n:],joint=joint[n:],
                            paid=paid[n:],fine=fine[n:],estimated_risk=pred)
        probability=np.where(change,fine[n:].mean(1),base)
        with np.load(model) as z:
            orig=z['first']
            trigger=(z['estimated_risk']>.5)&~orig&(z['joint']!=orig)&((z['parent']>=.5)==orig)
            np.testing.assert_array_equal(trigger,change)
            repeat=np.where(trigger,z['fine'].mean(1),z['parent'])
        np.testing.assert_array_equal(probability,repeat)
        pd.DataFrame(dict(PassengerId=test.PassengerId,Transported=probability>=.5)).to_csv(dest/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=test.PassengerId,probability=probability,reload_probability=repeat)).to_csv(dest/'test_probabilities.csv',index=False)
        check=validate_submission(dest/'submission.csv',ROOT/'data/raw/test.csv')
        assert not any(json.loads(f.read_text())['file_sha256']==check['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
        manifest=dict(**check,version=NAME.removeprefix('proposal_'),training_rows=n,model_sha256=dig(model),parents={CHAMP:dig(ART/CHAMP/'submission.csv'),JOINT:dig(ART/JOINT/'submission.csv')},reload_difference=0,changed=int(change.sum()),source_file='adjudicate_fifty_fare_n27.py')
        selection=dict(version=manifest['version'],rule=chosen,evidence=str(REPORT.relative_to(ROOT)),result=report[chosen],ordered=report['level_fifty_ordered_up'],fresh_order_warning='Only two additional correctly classified historical-unseen passengers in the random validation and zero edits in ordered validation.')
        for folder in [dest,ROOT/'reports'/NAME]:
            folder.mkdir(exist_ok=True);write(folder/'manifest.json',manifest);write(folder/'selection.json',selection)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export')
    with threadpool_limits(1):run(parser.parse_args().export)
