"""Cross-fit an all-passenger committee with three independent omissions of input families."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dissent_likelihood_transfer_n27 import splits
from export_budget_quarter_n23 import source
from repeated_archive_misses_n26 import ART,cohort,count
from unanimous_error_reach_n27 import ROOT,inputs
from kaggle_workflow import validate_submission,sha256
from relational_error_corrector import write

NAME='proposal_n28_complete_architecture_committee'
PARENT='proposal_n27_directional_foundation'
REPORT=ROOT/'reports/nov23_matched_learning_curve/complete_architecture_committee_n28.json'


def architectures(frame,y,f,wide,alt):
    ids=frame.PassengerId.to_numpy(str)
    def get(path,key):
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z['passenger'],ids)
            np.testing.assert_array_equal(z['y'],y)
            np.testing.assert_array_equal(z['fold'],f)
            return z[key]
    traditional=get('nov26_latent_passenger_difficulty/panel.npz','scores')[:,:18]
    structured=np.c_[get('nov27_field_pair_evidence/aligned.npz','random')[:,-1],
                     get('nov27_manifest_tree_mixture/scores.npz','random')[:,-1],
                     get('nov27_shared_interactions/predictions.npz','random')[:,-1],
                     get('nov27_manifest_activity_transitions/aligned.npz','random_asymmetric'),
                     get('nov27_solo_corridor_receipts/aligned.npz','random')[:,-1],
                     expit(get('nov27_conditional_party_allocation/aligned.npz','random_scores')[:,1])]
    old_alt=np.column_stack([alt[k] for k in ('bit_context','nearest','robust_boost','augmented_tree')])
    blocks=[traditional,wide[:,[1,2,3,5,6]],np.c_[structured,old_alt]]
    return [logit(p.clip(.005,.995)) for p in blocks]


def fit_predict(x,t,q,y,consensus):
    scale=StandardScaler().fit(x[t])
    tr=scale.transform(x[t]).clip(-7,7); qr=scale.transform(x[q]).clip(-7,7)
    # Partial pooling: the agreement group has its own correction to the common weights.
    tr=np.c_[tr,.4*tr*consensus[t,None],consensus[t]]
    qr=np.c_[qr,.4*qr*consensus[q,None],consensus[q]]
    model=LogisticRegression(C=.12,max_iter=450)
    return model.fit(tr,y[t]).decision_function(qr)


def run():
    raw,frame,y,g,f,first,edit,_,wide,alt=inputs()
    n=len(y); qtest=np.arange(n,len(frame))
    _, official,ph=source(PARENT,frame.iloc[n:])
    inc=first^edit
    np.testing.assert_array_equal(inc[n:],official)
    wrong=inc[:n]!=y; _,_,known,fresh=cohort(raw,y,f)
    same=((wide>=.5)==(wide[:,[0]]>=.5)).all(1)
    same_parent=same&((wide[:,0]>=.5)==inc)
    scopes={'all':known,'new':known&fresh,'seven_agree_parent':known&same_parent[:n],
            'seven_unanimously_oppose_parent':known&same[:n]&~same_parent[:n],
            'seven_divided':known&~same[:n]}
    blocks=architectures(frame,y,f,wide,alt)
    families=['all','no_classic','no_foundation','no_structured']
    designs=[np.column_stack([b for j,b in enumerate(blocks) if j!=omit]) for omit in (-1,0,1,2)]
    report=dict(baseline_correct=int((~wrong&known).sum()),known_rows=int(known.sum()),parent_hash=ph)
    saved={}
    for scheme in ('random','contiguous'):
        part=splits(g,f,scheme)
        z=np.full((len(frame),4),np.nan); test_votes=[]
        for k in np.unique(f):
            for l in (0,1,2,3,None):
                t=np.flatnonzero((f==k)&((part!=l) if l is not None else True))
                q=np.flatnonzero((f==k)&(part==l)) if l is not None else qtest
                if l is not None:assert not np.isin(g[t],g[q]).any()
                p=np.column_stack([fit_predict(d,t,q,y,same) for d in designs])
                if l is None:test_votes.append(p)
                else:z[q]=p
        z[n:]=np.mean(test_votes,axis=0)
        assert np.isfinite(z).all()
        dissent=(z>=0)!=inc[:,None]
        flags=dict(full=dissent[:,0],all_four=dissent.all(1),majority=dissent.sum(1)>=3)
        rec={}
        for name,flag in flags.items():
            change=flag[:n]*np.where(wrong,1,-1)
            rec[name]={k:count(flag[:n],m,wrong) for k,m in scopes.items()}
            rec[name].update(known_accuracy=float(np.mean((inc[:n]^flag[:n])[known]==y[known])),
                test=int(flag[n:].sum()),test_consensus_and_divided=[int((flag[n:]&s).sum()) for s in [same[n:],~same[n:]]],
                fold_net=[int(change[known&(f==k)].sum()) for k in np.unique(f)])
            saved[scheme+'_'+name]=flag
        saved[scheme+'_scores']=z
        report[scheme]=rec
    primary='all_four'
    assert np.array_equal(saved['random_'+primary][n:],saved['contiguous_'+primary][n:])
    testflag=saved['random_'+primary][n:]
    report['repeat']={k:count((saved['random_'+primary]&saved['contiguous_'+primary])[:n],m,wrong)
                      for k,m in scopes.items()}
    archive=ART/'nov28_complete_architecture_committee'
    archive.mkdir(exist_ok=True)
    np.savez_compressed(archive/'scores.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**saved)
    print({s:report[s][primary] for s in ['random','contiguous']},report['repeat'],flush=True)
    clear=all(report[s][primary]['known_accuracy']>.83 and
              report[s][primary]['all']['fixed']>report[s][primary]['all']['harmed'] for s in ['random','contiguous'])
    report['evidence_gate_met']=clear
    if clear and testflag.any():
        out=ART/NAME; out.mkdir(exist_ok=True)
        model=out/'inputs.npz'; ids=frame.PassengerId[n:].to_numpy(str)
        np.savez_compressed(model,parent=official,scores=saved['random_scores'][n:],passenger=ids)
        with np.load(model) as z:
            rebuilt=((z['scores']>=0)!=z['parent'][:,None]).all(1)
        np.testing.assert_array_equal(rebuilt,testflag)
        final=(official^rebuilt)
        parent_soft,_,_=source(PARENT,frame.iloc[n:])
        p=np.where(rebuilt,expit(saved['random_scores'][n:,0]),parent_soft)
        np.testing.assert_array_equal(p>=.5,final)
        pd.DataFrame(dict(PassengerId=ids,Transported=final)).to_csv(out/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=ids,probability=p,reload_probability=p)).to_csv(out/'test_probabilities.csv',index=False)
        checked=validate_submission(out/'submission.csv',ROOT/'data/raw/test.csv')
        samplings=[]
        for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
            rec=json.loads(path.read_text())
            if rec.get('evaluation',{}).get('Status')!='COMPLETE': continue
            file=ART/path.parents[1].name/'submission.csv'
            if file.exists() and sha256(file)==rec['file_sha256']:
                old=pd.read_csv(file); np.testing.assert_array_equal(old.PassengerId,ids)
                samplings.append([int(np.sum(final!=old.Transported)),path.parents[1].name])
        assert samplings and min(samplings)[0]>0
        report['nearest_submitted']=min(samplings)
        version=NAME.removeprefix('proposal_')
        for dest in [out,ROOT/'reports'/NAME]:
            write(dest/'manifest.json',dict(**checked,version=version,parents={PARENT:ph},model_sha256=sha256(model),reload_difference=0,
                                           edits=int(testflag.sum()),training_rows=n))
            write(dest/'selection.json',dict(version=version,rule='Four grouped readouts, including all three leave-one-family-out models, oppose the scored parent.'))
    write(REPORT,report)


if __name__=='__main__':
    with threadpool_limits(2):run()
