"""Compare broad joint predictions with three tied public releases, using held training labels only."""
import argparse
import json
import numpy as np
import pandas as pd
from export_consensus_review_n26 import ART,ROOT,write,dig
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import cohort,count
from six_way_against_scored_nine_n26 import test_decisions

FIRST='proposal_n27_directional_foundation'
RECEIPT='proposal_n27_receipt_mitra_party_union'
ROUTES='proposal_n27_three_payment_states'
ARCH='proposal_n27_architecture_increment'
NAME='proposal_n27_joint_error_experts'
REPORT=ROOT/'reports/nov23_matched_learning_curve/joint_repeated_experts.json'


def run(selected):
    raw,y,g,f=data();n=len(y)
    _,blind,known,fresh=cohort(raw,y,f)
    with np.load(ART/'nov27_repeated_architecture_increment/aligned.npz') as z:
        ids=z['passenger']
        np.testing.assert_array_equal(z['y'],y)
        architecture={k:z[k] for k in ('random','ordered')}
        parents={k:z[k+'_parent'] for k in architecture}
    with np.load(ART/'nov27_unsubmitted_audit/retained_party_fields_41.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        party=dict(random=z['random'],ordered=z['contiguous'])
        np.testing.assert_array_equal(parents['random'],z['random_parent'])
        np.testing.assert_array_equal(parents['ordered'],z['contiguous_parent'])
    with np.load(ART/'nov27_payment_state_routes/aligned.npz') as z:
        fixed=z['parent']
        routes={k:fixed^z[k].any(1) for k in parents}
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        first=z['first']
    with np.load(ART/'nov27_mitra_fare/aligned.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        mitra=z['111_specific_with_support']
    with np.load(ART/'nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        specialists={k:z[k] for k in parents}
    test=pd.Series(ids[n:])
    sources={k:test_decisions(k,test) for k in (FIRST,RECEIPT,ROUTES)}
    report={};test_arms={}
    for scheme,base in parents.items():
        with np.load(ART/'nov27_complete_receipt_counterfactual'/f'aligned_{scheme}_2.npz') as z:
            np.testing.assert_array_equal(z['ids'],ids)
            cf=z['average_random_calibrated']&z['average_ordered_calibrated']
        receipt=(mitra&(first==base))|((cf|specialists[scheme])&(fixed==base))
        np.testing.assert_array_equal((base^receipt)[n:],sources[RECEIPT][0])
        np.testing.assert_array_equal(routes[scheme][n:],sources[ROUTES][0])
        for name,action in [('arch_party',architecture[scheme]|party[scheme]),('receipt_arch',receipt|architecture[scheme]),
                            ('receipt_party',receipt|party[scheme]),('all_three',receipt|architecture[scheme]|party[scheme])]:
            candidate=base^action
            rows={}
            for par,decisions in [('directional',base),('receipt',base^receipt),('routes',routes[scheme])]:
                diff=candidate!=decisions
                err=decisions[:n]!=y;delta=np.where(err,1,-1)*diff[:n]
                rows[par]=dict(all=count(diff[:n],known,err),fresh=count(diff[:n],fresh&known,err),
                              common=count(diff[:n],blind,err),test=int(diff[n:].sum()),
                              by_fold=[int(delta[known&(f==j)].sum()) for j in range(8)])
            report.setdefault(name,{})[scheme]=rows
            if scheme=='random':test_arms[name]=candidate[n:]
            else:np.testing.assert_array_equal(test_arms[name],candidate[n:])
        arms=np.column_stack([receipt[n:],architecture[scheme][n:],party[scheme][n:]])
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,v in report.items():
        print(k,[(tag,r['directional']['test'],[(p,s['all']['fixed']-s['all']['harmed'],s['fresh']['fixed']-s['fresh']['harmed']) for p,s in r.items()]) for tag,r in v.items()])
    if selected is None:return
    flags=arms.copy()
    if selected!='all_three':
        skip={'arch_party':0,'receipt_party':1,'receipt_arch':2}[selected]
        flags[:,skip]=False
    def probabilities(label):
        frame=pd.read_csv(ART/label/'test_probabilities.csv',float_precision='round_trip')
        np.testing.assert_array_equal(frame.PassengerId,test)
        np.testing.assert_array_equal(frame.probability,frame.reload_probability)
        return frame.probability.to_numpy()
    p0=probabilities(FIRST)
    p1=probabilities(RECEIPT)
    p2=probabilities(ARCH)
    p3=np.column_stack([probabilities(x) for x in ['proposal_n25_confirmed_random_fields','proposal_n26_dispute_band']])
    valid=(p3>=.5)!=(p0>=.5)[:,None]
    historical=np.where(valid,p3,0.).sum(1)/valid.sum(1).clip(1)
    assert valid[flags[:,2]].any(1).all()
    scores=np.column_stack([p1,p2,historical])
    assert np.all(((scores>=.5)!=(p0>=.5)[:,None])[flags])
    path=ART/NAME;path.mkdir(exist_ok=True)
    model=path/'inputs.npz'
    np.savez_compressed(model,passenger=ids[n:],parent=p0,scores=scores,flags=flags)
    def apply(base,score,action):
        return np.where(action.any(1),np.where(action,score,0.).sum(1)/action.sum(1).clip(1),base)
    p=apply(p0,scores,flags)
    with np.load(model) as z: replay=apply(z['parent'],z['scores'],z['flags'])
    np.testing.assert_array_equal(p,replay)
    np.testing.assert_array_equal(p>=.5,test_arms[selected])
    csv=path/'submission.csv'
    pd.DataFrame({'PassengerId':ids[n:],'Transported':p>=.5}).to_csv(csv,index=False)
    pd.DataFrame({'PassengerId':ids[n:],'probability':p,'reload_probability':replay}).to_csv(path/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(f.read_text())['file_sha256']!=check['sha256'] for f in (ROOT/'reports').glob('*/online_attempts/*.json'))
    parenthash={k:sources[k][1] for k in [FIRST,RECEIPT]}
    m=dict(**check,version=NAME.removeprefix('proposal_'),parents=parenthash,training_rows=n,model_sha256=dig(model),
           reload_difference=0,changed_from_direction=int(np.sum((p>=.5)!=(p0>=.5))),
           changed_from_receipt=int(np.sum((p>=.5)!=(p1>=.5))),source_file='join_repeated_experts_n27.py')
    for folder in (path,ROOT/'reports'/NAME):
        write(folder/'manifest.json',m)
        write(folder/'selection.json',dict(version=m['version'],selected=selected,evidence=str(REPORT.relative_to(ROOT)),
              arms=['three receipt/party signals','TabDPT and TabNet increment','older fields and band on shared bookings']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',choices=['arch_party','receipt_arch','receipt_party','all_three'])
    run(parser.parse_args().export)
