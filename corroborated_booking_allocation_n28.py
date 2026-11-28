"""Preserve booking-allocation improvements and test corroborated reversal of a unanimous veto."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from complete_architecture_committee_n28 import ART,PARENT,ROOT,architectures,cohort,count,fit_predict,inputs,source,write
from kaggle_workflow import sha256,validate_submission

NAME='proposal_n28_corroborated_booking_allocation'
ONLY='proposal_n28_unanimous_architecture_opposition'


def final_decision(parent,scores,ordinary,allocated):
    opposite=((ordinary>=.5)!=parent[:,None]).all(1)
    corroborated=((scores>=0)!=parent[:,None]).all(1)
    return allocated|(opposite&corroborated)


def run():
    raw,frame,y,g,f,first,edits,_,wide,alt=inputs()
    n=len(y);test=np.arange(n,len(frame)); inc=first^edits
    soft,official,dig=source(PARENT,frame.iloc[n:])
    np.testing.assert_array_equal(inc[n:],official)
    wrong=inc[:n]!=y; _,_,known,fresh=cohort(raw,y,f)
    same=((wide>=.5)==(wide[:,[0]]>=.5)).all(1)
    oppose=((wide>=.5)!=inc[:,None]).all(1)
    blocks=architectures(frame,y,f,wide,alt)
    designs=[np.column_stack([b for j,b in enumerate(blocks) if j!=omit]) for omit in (-1,0,1,2)]
    report={}; saved={}; independent={}
    with np.load(ART/'nov27_conditional_party_allocation/aligned.npz') as cache:
        np.testing.assert_array_equal(cache['passenger'],frame.PassengerId.to_numpy(str))
        allocation={s:cache[key+'_supported_increment'] for s,key in [('random','random'),('contiguous','ordered')]}
    votes={}
    with np.load(ART/'nov28_complete_architecture_committee/scores.npz') as bank:
        for s in allocation:votes[s]=bank[s+'_scores']
    for seed in (28105,28107,28109):
        rng=np.random.default_rng(seed)
        order=np.unique(g); labels=rng.permutation(len(order))%4
        assignment=labels[np.searchsorted(order,g)]
        z=votes['random'].copy()
        # Other rows cannot activate the new rule even if all four readouts oppose the anchor.
        for k in np.unique(f):
            for part in range(4):
                t=np.flatnonzero((f==k)&(assignment!=part))
                q=np.flatnonzero((f==k)&(assignment==part)&oppose[:n])
                if not len(q):continue
                assert not np.isin(g[t],g[q]).any()
                z[q]=np.column_stack([fit_predict(x,t,q,y,same) for x in designs])
        votes[str(seed)]=z
    for s,z in votes.items():
        alloc=allocation['contiguous' if s=='contiguous' else 'random']
        act=final_decision(inc,z,wide,alloc)
        inc_edits=act&~alloc
        saved[s]=act
        alone=((wide>=.5)!=inc[:,None]).all(1)&((z>=0)!=inc[:,None]).all(1)
        retained=alone&alloc
        independent[s]=dict(all=count(alone[:n],known,wrong),fresh=count(alone[:n],known&fresh,wrong),
            fold_net=[int((alone[:n]*np.where(wrong,1,-1))[known&(f==k)].sum()) for k in np.unique(f)],
            accuracy=float(np.mean((inc[:n]^alone[:n])[known]==y[known])),tests=int(alone[n:].sum()),
            overlap_allocation_local=count(retained[:n],known,wrong),overlap_allocation_test=int(retained[n:].sum()))
        report[s]=dict(all=count(act[:n],known,wrong),fresh=count(act[:n],known&fresh,wrong),
            accuracy=float(np.mean((inc[:n]^act[:n])[known]==y[known])),
            incremental=count(inc_edits[:n],known,wrong),incremental_fresh=count(inc_edits[:n],known&fresh,wrong),
            agreement=count(act[:n],known&same[:n],wrong),disagreement=count(act[:n],known&~same[:n],wrong),
            tests=int(act[n:].sum()),extra_tests=int(inc_edits[n:].sum()),
            test_orders=int(frame.PassengerId.iloc[n:][act[n:]].str[:4].nunique()))
    solo_test=((wide[n:]>=.5)!=official[:,None]).all(1)&((votes['random'][n:]>=0)!=official[:,None]).all(1)
    score56=source('proposal_n27_conditional_party_allocation',frame.iloc[n:])[1]
    np.testing.assert_array_equal(score56!=official,allocation['random'][n:])
    comparison=[]
    ids=frame.PassengerId.to_numpy(str)[n:]
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        receipt=json.loads(path.read_text()); proposal=path.parents[1].name
        other=ART/proposal/'submission.csv'
        if receipt.get('evaluation',{}).get('Status')!='COMPLETE' or not other.exists(): continue
        if receipt['file_sha256']!=sha256(other):continue
        others=pd.read_csv(other);np.testing.assert_array_equal(others.PassengerId,ids)
        flags=others.Transported.to_numpy()!=official
        comparison.append(dict(version=proposal,distance=int(np.count_nonzero(flags^solo_test)),
            same_edits=int(np.count_nonzero(flags&solo_test)),score=receipt['evaluation']['Public Score']))
    independent['nearest']=sorted(comparison,key=lambda r:r['distance'])[:8]
    independent['max_edit_overlaps']=sorted(comparison,key=lambda r:(-r['same_edits'],r['distance']))[:8]
    checked_conditions=all(independent[s]['all']['fixed']>independent[s]['all']['harmed'] for s in votes)
    independent['exported']=bool(checked_conditions and solo_test.any() and independent['nearest'][0]['distance']>0)
    if independent['exported']:
        dest=ART/ONLY;dest.mkdir(exist_ok=True);model=dest/'inputs.npz'
        np.savez_compressed(model,passenger=ids,parent=official,soft=soft,seven=wide[n:],four=votes['random'][n:])
        with np.load(model) as z:
            a=((z['seven']>=.5)!=z['parent'][:,None]).all(1)&((z['four']>=0)!=z['parent'][:,None]).all(1)
            prob=np.where(a,1.-z['soft'],z['soft'])
            prob=np.where(a&(z['soft']==.5),.499999,prob)
        np.testing.assert_array_equal(prob>=.5,official^solo_test)
        csv=dest/'submission.csv'
        pd.DataFrame(dict(PassengerId=ids,Transported=prob>=.5)).to_csv(csv,index=False)
        pd.DataFrame(dict(PassengerId=ids,probability=prob,reload_probability=prob)).to_csv(dest/'test_probabilities.csv',index=False)
        valid=validate_submission(csv,ROOT/'data/raw/test.csv')
        for directory in [dest,ROOT/'reports'/ONLY]:
            write(directory/'manifest.json',dict(**valid,version=ONLY.removeprefix('proposal_'),model_sha256=sha256(model),
                training_rows=n,parents={PARENT:dig},reload_difference=0,changed=int(a.sum())))
            write(directory/'selection.json',dict(version=ONLY.removeprefix('proposal_'),
                rule='Only seven unanimous opposing votes confirmed after omitting each architectural family.',
                selected_from='Exploratory committee analysis. Extra readout splits re-use the historical passenger outcomes.',
                evidence='reports/nov23_matched_learning_curve/standalone_architecture_opposition_n28.json'))
    write(ROOT/'reports/nov23_matched_learning_curve/standalone_architecture_opposition_n28.json',independent)
    print(report,flush=True)
    for s,v in saved.items(): np.testing.assert_array_equal(v[n:],saved['random'][n:])
    report['exploration']='Unanimous opposition was identified after the first full-population committee. Extra seeds only repartition readout training on the same labeled passengers and base fits.'
    gate=report['random']['extra_tests']>0 and all(report[s]['incremental']['fixed']>report[s]['incremental']['harmed'] for s in votes)
    report['exported']=bool(gate)
    if gate:
        dest=ART/NAME; dest.mkdir(exist_ok=True)
        ids=frame.PassengerId.to_numpy(str)[n:]
        model=dest/'inputs.npz'
        np.savez_compressed(model,ids=ids,soft=soft,parent=official,ordinary=wide[n:],scores=votes['random'][n:],
                            allocated=allocation['random'][n:])
        with np.load(model) as pack:
            a=final_decision(pack['parent'],pack['scores'],pack['ordinary'],pack['allocated'])
            p=np.where(a,1.-pack['soft'],pack['soft'])
            p=np.where(a&(pack['soft']==.5),.499999,p)
        np.testing.assert_array_equal(p>=.5,official^saved['random'][n:])
        csv=dest/'submission.csv'
        pd.DataFrame(dict(PassengerId=ids,Transported=p>=.5)).to_csv(csv,index=False)
        pd.DataFrame(dict(PassengerId=ids,probability=p,reload_probability=p)).to_csv(dest/'test_probabilities.csv',index=False)
        check=validate_submission(csv,ROOT/'data/raw/test.csv')
        distances=[]
        for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
            rec=json.loads(path.read_text()); fpath=ART/path.parents[1].name/'submission.csv'
            if rec.get('evaluation',{}).get('Status')=='COMPLETE' and fpath.exists() and sha256(fpath)==rec['file_sha256']:
                b=pd.read_csv(fpath);np.testing.assert_array_equal(b.PassengerId,ids)
                distances.append([int((b.Transported!=(p>=.5)).sum()),path.parents[1].name])
        nearest=min(distances);assert nearest[0]>0
        version=NAME.removeprefix('proposal_')
        for folder in [dest,ROOT/'reports'/NAME]:
            write(folder/'manifest.json',dict(**check,version=version,training_rows=n,parents={PARENT:dig},model_sha256=sha256(model),
                  reload_difference=0,changed=int(a.sum())))
            write(folder/'selection.json',dict(version=version,rule='Existing booking allocation or seven unanimous opposite votes confirmed by all four leave-family-out readouts.',
                  nearest_scored=nearest,source='reports/nov23_matched_learning_curve/corroborated_booking_allocation.json'))
    write(ROOT/'reports/nov23_matched_learning_curve/corroborated_booking_allocation.json',report)


if __name__=='__main__':
    with threadpool_limits(2):run()
