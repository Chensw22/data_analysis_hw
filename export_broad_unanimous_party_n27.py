"""Freeze two independent party-expert increments on passengers with unanimous ordinary predictions."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from broad_population_expert_crosscheck_n27 import OUT,ART,REPORT,CHAMP,ROOT
from kaggle_workflow import validate_submission
from matched_state_context_n26 import inputs
from repeated_archive_misses_n26 import count,cohort

NAME='proposal_n27_unanimous_party_experts'


def ballot(parent,global_scores,experts,opinions):
    odds=lambda a:logit(np.clip(a,.004,.996))
    base=parent>=.5
    direction=np.where(base,-1.,1.)
    native=opinions[:,4]
    increment=(odds(experts)-odds(global_scores))*direction[:,None]
    unanimous=((opinions>=.5)==base[:,None]).all(1)
    change=unanimous&((experts>=.5)!=base[:,None]).all(1)&(increment>abs(odds(native))[:,None]).all(1)
    return change,np.where(change,experts.mean(1),parent)


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def run():
    frame,y,g,f,_=inputs();ids=frame.PassengerId.to_numpy(str);n=len(y)
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        outside=[np.median(z[k],axis=1) if z[k].ndim==2 else z[k] for k in ('dpt','tabnet')]
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        h=z['scores']
        opinions=np.column_stack([np.median(h[:,:18],axis=1),h[:,-3:],np.median(h[:,-3:],axis=1),*outside])
    scored=pd.read_csv(ART/CHAMP/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(scored.PassengerId,ids[n:])
    summary={}
    allacts=[]
    hard_scores=[]
    with np.load(OUT/'aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['y'],y)
        anchor=z['parent'];np.testing.assert_array_equal(anchor[n:],scored.probability>=.5)
        for mode,fold in [('random',f),('contiguous',z['ordered'])]:
            global_scores=z[mode+'_global']; experts=z[mode+'_party']
            a,decisions=ballot(1.*anchor,global_scores,experts,opinions)
            hard_scores.append(decisions>=.5)
            allacts.append(a)
            wrong=anchor[:n]!=y
            delta=a[:n]*np.where(wrong,1,-1)
            summary[mode]=dict(all=count(a[:n],known,wrong),fresh=count(a[:n],known&fresh,wrong),
                folds=[int(delta[(fold==j)&known].sum()) for j in range(8)],test=int(a[n:].sum()),
                precision=float(wrong[a[:n]&known].mean()))
    np.testing.assert_array_equal(allacts[0][n:],allacts[1][n:])
    np.savez_compressed(OUT/'unanimous_party_actions.npz',ids=ids,y=y,parent=anchor,opinions=opinions,random=allacts[0],ordered=allacts[1],
                        random_decisions=hard_scores[0],ordered_decisions=hard_scores[1],random_fold=f,ordered_fold=fold)
    selects=dict(random=[5,2],contiguous=[7,4])
    for k,expected in selects.items():
        assert [summary[k]['all']['fixed'],summary[k]['all']['harmed']]==expected
    folder=ART/NAME;folder.mkdir(exist_ok=True)
    packet=folder/'inputs.npz'
    np.savez_compressed(packet,passenger=ids[n:],parent=scored.probability.to_numpy(),global_scores=global_scores[n:],
                        experts=experts[n:],opinions=opinions[n:])
    with np.load(packet) as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:])
        a,p=ballot(z['parent'],z['global_scores'],z['experts'],z['opinions'])
    np.testing.assert_array_equal(a,allacts[0][n:])
    again=ballot(scored.probability.to_numpy(),global_scores[n:],experts[n:],opinions[n:])[1]
    np.testing.assert_array_equal(p,again)
    np.testing.assert_array_equal((p>=.5)!=anchor[n:],a)
    assert a.sum()==3==len(np.unique(g[n:][a]))
    pd.DataFrame({'PassengerId':ids[n:],'probability':p,'reload_probability':again}).to_csv(folder/'test_probabilities.csv',index=False)
    pd.DataFrame({'PassengerId':ids[n:],'Transported':p>=.5}).to_csv(folder/'submission.csv',index=False)
    check=validate_submission(folder/'submission.csv',ROOT/'data/raw/test.csv')
    assert not any(json.loads(path.read_text())['file_sha256']==check['sha256'] for path in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),training_rows=n,model_sha256=digest(packet),changed=int(a.sum()),
                  parents={CHAMP:digest(ART/CHAMP/'submission.csv')},reload_difference=0,source_file='export_broad_unanimous_party_n27.py')
    selection=dict(version=manifest['version'],rule='Among seven unanimous ordinary predictions, override when party and solo-specific gradient tree and logistic spline both favor the opposite label and each absolute log-odds change over its own global model exceeds the absolute median foundation log-odds.',
        current=CHAMP,metrics=summary,evidence=str(REPORT.relative_to(ROOT)),test_seven_probability_ranges=[
            {'passenger':str(pid),'lowest':float(values.min()),'highest':float(values.max()),'parent':bool(direction)}
            for pid,values,direction in zip(ids[n:][a],opinions[n:][a],anchor[n:][a])],
        local_validation='Both global and party/solo tree and spline architectures refit on each of eight random-order folds and eight contiguous-order folds; original seven opinion sources and scored parent unchanged between protocols.',
        limitations='Exploratory selection from 30 declared policy comparisons. The seven ordinary OOF outputs and the anchor are fixed across both protocols. The new expert architectures are independently refitted in both splits; 1,033 historically unseen passengers are 1 fixed and 1 harmed in each.',test_changed=ids[n:][a].tolist())
    for dest in [folder,ROOT/'reports'/NAME]:
        dest.mkdir(exist_ok=True)
        for key,content in [('manifest',manifest),('selection',selection)]:
            (dest/(key+'.json')).write_text(json.dumps(content,indent=2)+'\n')
    print(manifest,summary,ids[n:][a])

if __name__=='__main__':
    run()
