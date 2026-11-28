"""Estimate how often corroborated exceptions are correct, on other orders in the same base-model fold."""
import json
import numpy as np

from conditional_shortcut_boost_n28 import ART,ROOT,count,experiment_inputs
from dissent_likelihood_transfer_n27 import splits
from unanimous_error_reach_n27 import inputs


def features(parent,coarse,ordinary,exception,balanced):
    # Coarse direction and ordinary-model disagreement form the matched baseline strata.
    base=parent.astype(int)*4+((coarse>=0)!=parent)*2+((ordinary>=0)!=parent)
    pattern=((exception>=0)!=parent)*2+((balanced>=0)!=parent)
    return base,pattern


def estimate(base,pattern,t,q,wrong):
    counts=np.bincount(base[t],minlength=8)
    fixes=np.bincount(base[t],weights=wrong[t],minlength=8)
    broad=(fixes+1)/(counts+2)
    cells=base*4+pattern
    sample=np.bincount(cells[t],minlength=32)
    errors=np.bincount(cells[t],weights=wrong[t],minlength=32)
    # Partial pooling keeps behavior on sparsely represented patterns close to the baseline.
    fine=(errors+8*np.repeat(broad,4))/(sample+8)
    return broad[base[q]],fine[cells[q]]


def run():
    frame,y,f,parent,groups,_,_,known,_,wrong,masks,_,_=experiment_inputs()
    # On common rows both baselines equal the seven-way direction; allow the decision threshold
    # to differ between agreement and disagreement populations through separate training banks.
    opinions=inputs()[-2]
    agree=((opinions>=.5)==parent[:,None]).all(1)
    n=len(y)
    with np.load(ART/'nov28_conditional_shortcut_boost/aligned.npz') as a,np.load(ART/'nov28_retrospective_manifest_risk/aligned.npz') as b:
        for z in [a,b]:
            np.testing.assert_array_equal(z['passenger'],frame.PassengerId)
            np.testing.assert_array_equal(z['y'],y)
            np.testing.assert_array_equal(z['fold'],f)
        coarse,ordinary,offset=a['random'][:,:3].T
        balanced=b['random'][:,1]
    base,pattern=features(parent,coarse,ordinary,offset,balanced)
    report,saved={},{}
    for scheme in ('random','contiguous'):
        part=splits(groups[:n],f,scheme)
        decisions=np.zeros((len(frame),2),bool)
        for population in [False,True]:
            test=np.flatnonzero(agree[n:]==population)+n
            ballots=[]
            for k in range(8):
                bank=known&(f==k)&(agree[:n]==population)
                for held in [0,1,2,3,None]:
                    t=np.flatnonzero(bank if held is None else bank&(part!=held))
                    q=test if held is None else np.flatnonzero(bank&(part==held))
                    assert not np.isin(groups[t],groups[q]).any()
                    general,specific=estimate(base,pattern,t,q,wrong)
                    vote=np.column_stack([general>.5,specific>.5]) & (pattern[q,None]>0)
                    if held is None:
                        ballots.append(vote)
                    else:
                        decisions[q]=vote
            decisions[test]=np.mean(ballots,axis=0)>.5
        rules=dict(base=decisions[:,0],paired=decisions[:,1],new_only=decisions[:,1]&~decisions[:,0])
        report[scheme]={}
        for key,flag in rules.items():
            saved[scheme+'_'+key]=flag
            report[scheme][key]=dict(scopes={s:count(flag[:n],m,wrong) for s,m in masks.items()},
                accuracy=float(np.mean((parent[:n]^flag[:n])[known]==y[known])),test=int(flag[n:].sum()))
        print(scheme,report[scheme],flush=True)
    for key in rules:
        np.testing.assert_array_equal(saved['random_'+key][n:],saved['contiguous_'+key][n:])
    out=ART/'nov28_joint_exception_frequency'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'aligned.npz',passenger=frame.PassengerId.to_numpy(str),y=y,fold=f,**saved)
    (ROOT/'reports/nov23_matched_learning_curve/joint_exception_frequency.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
