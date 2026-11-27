"""Choose between two scored decisions using only independently computed validation views."""
import argparse
import json
import numpy as np
import pandas as pd
from export_consensus_review_n26 import ART, ROOT, dig, write
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import cohort, count
from unanimous_error_reach_n27 import inputs
from six_way_against_scored_nine_n26 import test_decisions

LEFT='proposal_n27_directional_foundation'
RIGHT='proposal_n27_three_payment_states'
NAME='proposal_n27_independent_payment_arbitration'
REPORT=ROOT/'reports/nov23_matched_learning_curve/scored_payment_arbitration.json'


def run(export):
    raw,frame,y,groups,fold,first,random,ordered,wide,alt=inputs()
    n=len(y); ids=frame.PassengerId.to_numpy(str)
    _,blind,known,fresh=cohort(raw,y,fold)
    with np.load(ART/'nov27_payment_state_routes/aligned.npz') as z:
        for k,val in [('passenger',ids),('y',y),('fold',fold)]:np.testing.assert_array_equal(z[k],val)
        base=z['parent']
        route={k:z[k].any(1) for k in ('random','ordered')}
    a,sha_a=test_decisions(LEFT,pd.Series(ids[n:]))
    b,sha_b=test_decisions(RIGHT,pd.Series(ids[n:]))
    np.testing.assert_array_equal(base[n:],a)
    assert (a!=b).sum()==68
    for flag in route.values():np.testing.assert_array_equal(flag[n:],a!=b)
    # Separate views: pretrained transformer, a tree fitted directly on amounts, and a class-conditional
    # geometric model. None used public test outcomes when trained or when producing these scores.
    vs=np.column_stack([wide[:,5],alt['same_input_tree'],alt['hull']])
    oppose=(vs>=.5)!=base[:,None]
    regs={'majority':oppose.sum(1)>=2,'all_three':oppose.all(1),
          'transformer_tree':oppose[:,0]&oppose[:,1], 'tree_geometry':oppose[:,1]&oppose[:,2],
          'transformer_geometry':oppose[:,0]&oppose[:,2], 'tree':oppose[:,1],
          'transformer':oppose[:,0], 'geometry':oppose[:,2]}
    report={}; selected=None
    for name,permission in regs.items():
        for tag,flag in route.items():
            adopt=permission&flag
            candidate=base^adopt
            against_b=flag&~permission
            err=base[:n]!=y
            err_b=(base^flag)[:n]!=y
            delta=np.where(err,1,-1)*adopt[:n]
            report.setdefault(name,{})[tag]=dict(all=count(adopt[:n],known,err),fresh=count(adopt[:n],known&fresh,err),
                old=count(adopt[:n],blind,err),vs_second=count(against_b[:n],known,err_b),
                fresh_vs_second=count(against_b[:n],known&fresh,err_b),by_fold=[int(delta[known&(fold==k)].sum()) for k in range(8)],
                test=int(adopt[n:].sum()),test_vs_second=int(against_b[n:].sum()))
            if name==export:selected=candidate[n:]
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for name,d in report.items():print(name,[(v['all'],v['vs_second'],v['fresh'],v['test']) for v in d.values()])
    if export is None:return
    assert selected is not None
    path=ART/NAME
    path.mkdir(exist_ok=True)
    def scores(tag):
        v=pd.read_csv(ART/tag/'test_probabilities.csv',float_precision='round_trip')
        np.testing.assert_array_equal(v.PassengerId,ids[n:])
        np.testing.assert_array_equal(v.probability,v.reload_probability)
        return v.probability.to_numpy()
    par,other=scores(LEFT),scores(RIGHT)
    source=path/'inputs.npz'
    np.savez_compressed(source,passenger=ids[n:],left=par,right=other,permit=regs[export][n:],views=vs[n:])
    p=np.where(regs[export][n:],other,par)
    with np.load(source) as z:r=np.where(z['permit'],z['right'],z['left'])
    np.testing.assert_array_equal(p,r)
    np.testing.assert_array_equal(p>=.5,selected)
    csv=path/'submission.csv'
    pd.DataFrame({'PassengerId':ids[n:],'Transported':p>=.5}).to_csv(csv,index=False)
    pd.DataFrame({'PassengerId':ids[n:],'probability':p,'reload_probability':r}).to_csv(path/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(f.read_text())['file_sha256']!=check['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    m=dict(**check,version=NAME.removeprefix('proposal_'),parents={LEFT:sha_a,RIGHT:sha_b},training_rows=n,
           changed=int((selected!=a).sum()),changed_from_second=int((selected!=b).sum()),model_sha256=dig(source),reload_difference=0,
           source_file='arbitrate_scored_payments_n27.py')
    for folder in (path,ROOT/'reports'/NAME):
        write(folder/'manifest.json',m)
        write(folder/'selection.json',dict(version=m['version'],selected=export,compared=list(regs),evidence=str(REPORT.relative_to(ROOT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export')
    run(parser.parse_args().export)
