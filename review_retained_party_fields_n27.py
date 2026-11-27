"""Replay the broad party correction and audit its overlap with current candidates."""
import json

import numpy as np
import pandas as pd

from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from six_way_against_scored_nine_n26 import test_decisions
from frozen_queue_shared_errors_n26 import RELEASES
from audit_unsubmitted_branches_n27 import FIRST,BEST,OUT,digest

REPORT=ROOT/'reports/nov23_matched_learning_curve/retained_party_fields_overlap.json'


def run():
    raw,y,g,f=data();n=len(y)
    _,blind,known,fresh=cohort(raw,y,f)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)
    frame=pd.concat([raw,test],ignore_index=True);ids=frame.PassengerId.to_numpy(str)
    groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    multi=pd.Series(groups).duplicated(keep=False).to_numpy()
    with np.load(OUT/'archived_releases.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['y'],y)
        bank={name:z['pred'][:,i] for i,name in enumerate(z['names'])}
    for name in ['nine','fields','band']:
        np.testing.assert_array_equal(bank[name][n:],test_decisions(RELEASES[name],test.PassengerId)[0])
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        first=z['first'];r=first^z['random_median'];c=first^z['contiguous_median']
    np.testing.assert_array_equal(first[n:],test_decisions(FIRST,test.PassengerId)[0])
    np.testing.assert_array_equal(r[n:],test_decisions(BEST,test.PassengerId)[0])
    with np.load(ART/'nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        opinions=z['opinions']
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        neural=[z['spread_split_anchored_random'],z['spread_split_anchored_ordered']]
    prev26=test_decisions('proposal_n27_unpaid_group_transfer',test.PassengerId)[0]
    small=test_decisions('proposal_n27_receipt_mitra_party_union',test.PassengerId)[0]
    official45=test_decisions('proposal_n27_architecture_increment',test.PassengerId)[0]
    p=(bank['fields']!=bank['nine'])|(bank['band']!=bank['nine'])
    gate=multi & (bank['nine']==first) & p
    report={};arrays={}
    for scheme,b,z in zip(['random','contiguous'],[r,c],neural):
        flag=gate&(first==b)
        # The archived neural directions are defined relative to 111. When the parent is recomputed on a new
        # calibration partition, protect those newly confirmed incumbent changes as for the party branch.
        alt=z&(first==b)
        assert np.array_equal(b[n:]^alt[n:],official45)
        seven=known&((opinions[:n]>=.5)==b[:n,None]).all(1)
        scopes=dict(all=known,fresh=known&fresh,seven=seven,old31=blind)
        results={}
        for label,interest in [('party',flag),('neural',alt),('union',flag|alt),('overlap',flag&alt)]:
            d=interest[:n]*np.where(b[:n]!=y,1,-1)
            results[label]=dict(scopes={k:count(interest[:n],m,b[:n]!=y) for k,m in scopes.items()},
                folds=[int(d[known&(f==k)].sum()) for k in range(8)],test=int(interest[n:].sum()))
        arrays[scheme]=flag
        arrays[scheme+'_parent']=b
        arrays[scheme+'_seven']=seven
        report[scheme]=results
    np.testing.assert_array_equal(arrays['random'][n:],arrays['contiguous'][n:])
    flag=arrays['random'][n:]
    pred=r[n:]^flag
    assert flag.sum()==41
    releases={'previous_26_vs_111':prev26!=first[n:],'previous_26_vs_16':prev26!=r[n:],
              'neural45_vs_16':official45!=r[n:],'three_branch18_vs_16':small!=r[n:]}
    report['test_overlap']={k:dict(other_edits=int(q.sum()),intersection=int((q&flag).sum()),union=int((q|flag).sum())) for k,q in releases.items()}
    report['test_overlap']['sha256']=digest(ids[n:],pred)
    # Repeat with exactly the archived recalibrations of the 45-person release, including its own recalibrated parent.
    replication=ART/'nov27_repeated_architecture_increment/aligned.npz'
    with np.load(replication) as z:
        np.testing.assert_array_equal(ids,z['passenger']);np.testing.assert_array_equal(y,z['y']);np.testing.assert_array_equal(f,z['fold'])
        rows={}
        for label in [key[:-7] for key in z.files if key.endswith('_parent')]:
            b=z[label+'_parent'];a=z[label];v=gate&(first==b)
            np.testing.assert_array_equal(flag,v[n:]);np.testing.assert_array_equal(official45,b[n:]^a[n:])
            seven=known&((opinions[:n]>=.5)==b[:n,None]).all(1)
            rows[label]={key:{s:count(action[:n],m,b[:n]!=y) for s,m in [('all',known),('fresh',known&fresh),('seven',seven),('old31',blind)]}
                          for key,action in [('party',v),('neural',a),('union',v|a)]}
            arrays['replicate_'+label+'_party']=v
            arrays['replicate_'+label+'_union']=v|a
        report['matched_neural_recalibrations']=rows
    np.savez_compressed(OUT/'retained_party_fields_41.npz',ids=ids,y=y,fold=f,first=first,test_ids=ids[n:],test_flag=flag,
              test_prediction=pred,**arrays)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(report)
if __name__=='__main__':run()
