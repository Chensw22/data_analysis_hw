"""Test the strength of three independent fare readers where the original six-way vote lacked unanimity."""
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from export_directional_foundation_n27 import NAME as CHAMP
from export_joint_fare_foundations_n26 import BASE,NAME as PAIR
from export_consensus_review_n26 import ART,ROOT,write,dig
from rebase_fare_six_n26 import predict
from repeated_archive_misses_n26 import cohort,count
from kaggle_workflow import validate_submission
from full_roster_booking_validation_n23 import data
from held_expert_portfolios_n26 import SIX,sources,panel_predictions
from arbitration_against_scored_nine_n26 import champions
from price_implied_deck_n26 import build_features,OUT as DECK
from nori_fare_party_n26 import ART as NORI
from nori_booking_regression_n26 import packet_dir
from solo_selection_likelihood_n26 import fit_model
from six_way_against_scored_nine_n26 import test_decisions

OUT=ART/'nov27_incomplete_six_fare_strength'
REPORT=ROOT/'reports/nov23_matched_learning_curve/incomplete_six_fare_strength.json'
NAME='proposal_n27_incomplete_six_fare_strength'


def strength(nine,six,before,after):
    # Each contributor has range zero to one, with one half denoting neutral evidence.
    opposition=((six>=.5)!=nine[:,None]).sum(1)
    support=np.where(nine[:,None],1-after,after)
    raw=np.where(nine[:,None],1-before,before)
    fraction=(opposition+.5)/7
    scores=dict(receipt=(fraction+support.mean(1)+(.5+.5*(support-raw).mean(1)))/3,
                raw_control=(fraction+raw.mean(1)+.5)/3)
    return opposition,scores


def load():
    frame,y,g,f=data()
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=frame.columns)
    all_frame=pd.concat([frame,test],ignore_index=True)
    ids=all_frame.PassengerId.to_numpy(str)
    n=len(y)
    nine,known,_,_=champions()
    v=sources()[-1]['full']
    p=panel_predictions(v,'full')
    six=np.column_stack([p[k] for k in SIX])
    test_ids=test.PassengerId
    with np.load(ART/BASE/'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test_ids)
        old=np.r_[nine,z['anchor']]
        six=np.vstack([six,z['challengers']])
    with np.load(ART/PAIR/'inputs.npz') as z:
        np.testing.assert_array_equal(z['passenger'],test_ids)
        test_before,test_after=z['raw'],z['fine']
    nr=np.empty(n);nf=nr.copy()
    for j in range(8):
        t,q=np.flatnonzero(f!=j),np.flatnonzero(f==j)
        for dest,path in [(nr,packet_dir(0)/f'fold{j}.npz'),(nf,NORI/f'own{j}.npz')]:
            with np.load(path) as z:
                np.testing.assert_array_equal(z['t'],t)
                np.testing.assert_array_equal(z['q'],q)
                np.testing.assert_array_equal(z['labels'],y[t])
                dest[q]=z['p']
    features=build_features(all_frame,all_frame.PassengerId.str[:4].to_numpy())
    path=OUT/'tree_test.npz'
    if not path.exists():
        result={}
        with np.load(DECK/'tree_full_0.npz') as z:
            for name in ('raw','fine'):
                x=features[name]
                refit=fit_model(x.iloc[z['t']],y[z['t']]).predict_proba(x.iloc[z['q']])[:,1]
                np.testing.assert_allclose(refit,z[name],atol=1e-8,rtol=0)
                result[name]=fit_model(x.iloc[:n],y).predict_proba(x.iloc[n:])[:,1]
        np.savez_compressed(path,passenger=ids[n:],labels=y,**result)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['passenger'],test_ids)
        np.testing.assert_array_equal(z['labels'],y)
        before=np.vstack([np.column_stack([p['neural_deck_raw'],nr,p['tree_deck_raw']]),np.column_stack([test_before,z['raw']])])
        after=np.vstack([np.column_stack([p['neural_deck_fine'],nf,p['tree_deck_fine']]),np.column_stack([test_after,z['fine']])])
    with np.load(DECK/'deck_estimates.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        eligible=z['eligible']
    with np.load(ART/'nov26_nori_fare_party/aligned.npz') as z: first=z['champion111'].astype(bool)
    first=np.r_[first,test_decisions(BASE,test_ids)[0]]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        med=np.median(z['scores'][:n,-3:],axis=1)
    current=first.copy()
    part=partition(g,'directional_foundation',4)
    risk=np.where(first[:n],1-med,med)
    for j in range(8):
        for m in range(4):
            tr=np.flatnonzero(known&(f==j)&(part!=m)); q=np.flatnonzero((f==j)&(part==m))
            cut=calibrate(risk,first,y,tr,1.)
            current[q]^=risk[q]>=cut[first[q].astype(int)]
    current[n:]=test_decisions(CHAMP,test_ids)[0]
    _,blind,known2,fresh=cohort(frame,y,f)
    np.testing.assert_array_equal(known,known2)
    return ids,y,g,f,old,first,current,six,before,after,eligible,known,blind,fresh


def cutoff(scores,truth,base,rows,separate):
    # Abstain if the cumulative gain less a square-root penalty stays nonpositive.
    output=np.full(2,np.inf)
    for side in ([False,True] if separate else [None]):
        r=rows if side is None else rows[base[rows]==side]
        r=r[scores[r]>.5]
        r=r[np.argsort(-scores[r],kind='stable')]
        if not len(r):continue
        gain=np.cumsum(np.where(truth[r]!=base[r],1,-1))-np.sqrt(np.arange(1,len(r)+1))
        gain[~np.r_[scores[r][:-1]!=scores[r][1:],True]]=-np.inf
        if gain.max()>0:
            val=scores[r[gain.argmax()]]
            if side is None:output[:]=val
            else:output[int(side)]=val
    return output


def run(export=None):
    OUT.mkdir(exist_ok=True)
    ids,y,g,f,nine,first,current,six,before,after,paid,known,blind,fresh=load()
    n=len(y)
    opp,scores=strength(nine,six,before,after)
    scope=paid&(opp<6)&(first==nine)
    apply_scope=scope&(current==nine)
    cache={};report={};cuts={}
    for name,risk in scores.items():
        for policy in ['shared','directional']:
            key=name+'_'+policy
            c=np.stack([cutoff(risk,y,nine,np.flatnonzero((f==j)&known&scope[:n]),policy=='directional') for j in range(8)])
            cuts[key]=c
            test=apply_scope[n:]&((risk[n:,None]>=c[:,nine[n:].astype(int)].T).sum(1)>4)
            result={}
            for seed in ['ordered','044','161','934']:
                part=partition(g,'incomplete_paid_'+seed,4) if seed!='ordered' else np.zeros(n,int)
                action=np.zeros(n,bool)
                for j in range(8):
                    if seed=='ordered':
                        for m,groups in enumerate(np.array_split(np.unique(g[f==j]),4)):
                            part[(f==j)&np.isin(g,groups)]=m
                    for m in range(4):
                        t=np.flatnonzero(known&scope[:n]&(f==j)&(part!=m))
                        q=np.flatnonzero(apply_scope[:n]&(f==j)&(part==m))
                        assert not np.isin(g[t],g[q]).any()
                        cut=cutoff(risk,y,nine,t,policy=='directional')
                        action[q]=risk[q]>=cut[nine[q].astype(int)]
                cache[key+'_'+seed]=action
                delta=action*np.where(current[:n]!=y,1,-1)
                result[seed]=dict(folds=[int(delta[known&(f==j)].sum()) for j in range(8)],
                    scopes={k:count(action,a,current[:n]!=y) for k,a in [('known',known),('fresh',known&fresh),('blind',blind),
                            ('negative',known&~current[:n]),('positive',known&current[:n])]})
            report[key]=dict(test_changes=int(test.sum()),test_count_by_votes=np.bincount(opp[n:][test],minlength=7).tolist(),validation=result)
            cache[key+'_test']=test
    np.savez_compressed(OUT/'choices.npz',ids=ids,y=y,fold=f,before=before,after=after,six=six,parent=current,nine=nine,scope=apply_scope,**cache)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():
        print(k,v['test_changes'],[(s,d['scopes']['known'],d['scopes']['fresh']) for s,d in v['validation'].items()])
    if export is None:return
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    champ=pd.read_csv(ART/CHAMP/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(champ.PassengerId,ids[n:])
    parent=champ.probability.to_numpy()
    np.testing.assert_array_equal(parent>=.5,current[n:])
    inputs=dict(parent=parent,nine=nine[n:],eligible=apply_scope[n:],six=six[n:],before=before[n:],after=after[n:],cuts=cuts[export])
    def rebuild(parent,nine,eligible,six,before,after,cuts):
        p=strength(nine,six,before,after)[1]['receipt']
        change=eligible&((p[:,None]>=cuts[:,nine.astype(int)].T).sum(1)>4)
        return np.where(change,np.where(nine,1-p,p),parent)
    model=dest/'inputs.npz'
    prob=rebuild(**inputs)
    np.testing.assert_array_equal((prob>=.5)!=current[n:],cache[export+'_test'])
    np.savez_compressed(model,passenger=ids[n:],**inputs)
    with np.load(model) as z:replay=rebuild(**{k:z[k] for k in inputs})
    np.testing.assert_array_equal(prob,replay)
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=prob>=.5)).to_csv(dest/'submission.csv',index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=prob,reload_probability=replay)).to_csv(dest/'test_probabilities.csv',index=False)
    checked=validate_submission(dest/'submission.csv',ROOT/'data/raw/test.csv')
    assert all(json.loads(fl.read_text())['file_sha256']!=checked['sha256'] or fl.parents[1].name==NAME for fl in (ROOT/'reports').glob('*/online_attempts/*.json'))
    m=dict(**checked,version='n27_incomplete_six_fare_strength',training_rows=n,parents={CHAMP:dig(ART/CHAMP/'submission.csv'),BASE:dig(ART/BASE/'submission.csv')},
           source_file='incomplete_six_fare_strength_n27.py',model_sha256=dig(model),reload_difference=0,changed=report[export]['test_changes'])
    s=dict(version=m['version'],rule=export,evidence=str(REPORT.relative_to(ROOT)))
    for folder in [dest,ROOT/'reports'/NAME]:
        folder.mkdir(exist_ok=True);write(folder/'manifest.json',m);write(folder/'selection.json',s)
    print(m)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',choices=['receipt_shared','receipt_directional'])
    with threadpool_limits(2):run(parser.parse_args().export)
