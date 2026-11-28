"""Transfer only the change in evidence from new architectures into the directional incumbent."""
import numpy as np
from scipy.special import logit
from complete_architecture_committee_n28 import ART,ROOT,PARENT,inputs,cohort,count,source,write
import pandas as pd
from kaggle_workflow import sha256,validate_submission

NAME='proposal_n28_architecture_increment'


def flags(z,a,ordinary,allocation):
    # A registered change must cross the frozen parent's foundation margin and oppose its label.
    s=np.where(a,1.,-1.)*np.maximum(1e-8,abs(logit(ordinary.clip(.003,.997))))
    full=z[:,0]>=0
    results={}
    for j,name in [(1,'classic_increment'),(2,'foundation_increment'),(3,'structured_increment')]:
        results[name]=(full!=a)&((s+z[:,0]-z[:,j]>=0)!=a)
    results['old_plus_new_increment']=(full!=a)&((s+(z[:,0]-z[:,1])+(z[:,0]-z[:,3])>=0)!=a)
    # Each branch proposes a reversal of the same parent vote.
    results['allocation_union']=allocation|results['structured_increment']
    return results


def run():
    raw,frame,y,g,f,first,edits,_,wide,_=inputs()
    n=len(y); inc=first^edits
    p0,official,digest=source(PARENT,frame.iloc[n:])
    np.testing.assert_array_equal(inc[n:],official)
    _,_,known,fresh=cohort(raw,y,f)
    wrong=inc[:n]!=y; ordinary=np.median(wide[:,1:4],axis=1)
    anycommon=((wide>=.5)==(wide[:,[0]]>=.5)).all(1)
    scopes=dict(all=known,fresh=known&fresh,agree=known&anycommon[:n],divided=known&~anycommon[:n])
    inputs_dir=ART/'nov28_complete_architecture_committee'
    report={};saved={}
    for scheme in ('random','contiguous'):
        with np.load(inputs_dir/'scores.npz') as z:
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId.to_numpy(str))
            np.testing.assert_array_equal(z['fold'],f);np.testing.assert_array_equal(z['y'],y)
            logits=z[scheme+'_scores']
        with np.load(ART/'nov27_conditional_party_allocation/aligned.npz') as z:
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId.to_numpy(str))
            allocation=z[('random' if scheme=='random' else 'ordered')+'_supported_increment']
        plans=flags(logits,inc,ordinary,allocation); plans['allocation_only']=allocation
        record={}
        for name,action in plans.items():
            saved[scheme+'_'+name]=action
            record[name]={k:count(action[:n],m,wrong) for k,m in scopes.items()}
            record[name].update(test=int(action[n:].sum()),consensus_test=int((anycommon[n:]&action[n:]).sum()),
              known_accuracy=float(np.mean((inc[:n]^action[:n])[known]==y[known])),
              additions_over_allocation=count((action&~allocation)[:n],known,wrong),
              test_additions_over_allocation=int((action&~allocation)[n:].sum()))
        report[scheme]=record; print(scheme,record,flush=True)
    selected='allocation_union'
    held=saved['random_'+selected]&saved['contiguous_'+selected]
    report['repeat']={k:count(held[:n],m,wrong) for k,m in scopes.items()}
    newcomer=held&~(saved['random_allocation_only']|saved['contiguous_allocation_only'])
    report['repeat_new_over_allocation']={k:count(newcomer[:n],m,wrong) for k,m in scopes.items()}
    delta=[report[s][selected]['additions_over_allocation'] for s in ['random','contiguous']]
    green=all(c['fixed']>c['harmed'] for c in delta) and all(report[s][selected]['known_accuracy']>.83 for s in ['random','contiguous'])
    report['exported']=green
    np.savez_compressed(inputs_dir/'increments.npz',passenger=frame.PassengerId.to_numpy(str),fold=f,y=y,**saved)
    if green:
        dest=ART/NAME;dest.mkdir(exist_ok=True)
        q=np.arange(n,len(frame)); archive=dest/'inputs.npz'
        np.savez_compressed(archive,passenger=frame.PassengerId.to_numpy(str)[q],parent=official,probability=p0,
                            scores=logits[q],ordinary=ordinary[q],allocation=allocation[q])
        with np.load(archive) as z:
            act=flags(z['scores'],z['parent'],z['ordinary'],z['allocation'])[selected]
            prob=np.where(act,1.-z['probability'],z['probability'])
            prob=np.where(act & (z['probability']==.5),.499999,prob)
        expected=(official^saved['random_'+selected][q])
        np.testing.assert_array_equal(expected,prob>=.5)
        ids=frame.PassengerId.to_numpy(str)[q]
        csv=dest/'submission.csv'
        pd.DataFrame(dict(PassengerId=ids,Transported=expected)).to_csv(csv,index=False)
        pd.DataFrame(dict(PassengerId=ids,probability=prob,reload_probability=prob)).to_csv(dest/'test_probabilities.csv',index=False)
        checked=validate_submission(csv,ROOT/'data/raw/test.csv')
        for d in [dest,ROOT/'reports'/NAME]:
            write(d/'manifest.json',dict(**checked,version=NAME[9:],changed=int(act.sum()),model_sha256=sha256(archive),reload_difference=0,
                parents={PARENT:digest},training_rows=n))
            write(d/'selection.json',dict(version=NAME[9:],rule='Conditional allocation union with shared structural evidence increment.'))
    write(ROOT/'reports/nov23_matched_learning_curve/architecture_odds_increment_n28.json',report)


if __name__=='__main__':run()
