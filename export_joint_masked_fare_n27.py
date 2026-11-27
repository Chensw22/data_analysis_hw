"""Evaluate and freeze the union of the disjoint complete-fare and incomplete-bill decisions."""
import json
import numpy as np
import pandas as pd

from adjudicate_fifty_fare_n27 import NAME as FARE,STORE
from export_consensus_review_n26 import ART,ROOT,dig,write
from export_directional_foundation_n27 import NAME as BASE
from export_matched_receipt_context_n27 import NAME as PARTIAL
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from masked_reference_receipts_n27 import OUT,FEES
from repeated_archive_misses_n26 import cohort,count

NAME='proposal_n27_joint_masked_fare'
REPORT=ROOT/'reports/nov23_matched_learning_curve/joint_masked_fare.json'


def apply(parent,fare,partial,incomplete):
    sides=parent>=.5
    a=(fare>=.5)!=sides
    b=(partial>=.5)!=sides
    assert not (a&b).any() and not (a&incomplete).any() and not (b&~incomplete).any()
    return np.where(a,fare,np.where(b,partial,parent))


def run():
    raw,y,g,f=data()
    _,blind,known,fresh=cohort(raw,y,f)
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    n=len(y)
    ids=np.r_[raw.PassengerId.to_numpy(str),test.PassengerId.to_numpy(str)]
    incomplete=np.r_[raw[FEES].isna().any(axis=1),test[FEES].isna().any(axis=1)]
    with np.load(OUT/'replica/aligned.npz') as za,np.load(STORE/'scores.npz') as zb:
        for field,q in [('y',y),('fold',f)]:
            np.testing.assert_array_equal(za[field],q);np.testing.assert_array_equal(zb[field],q)
        np.testing.assert_array_equal(za['passenger'],ids);np.testing.assert_array_equal(zb['ids'],ids)
        parent=zb['parent']
        np.testing.assert_array_equal(za['anchor'],parent)
        missing=za['action_replicate_independent_shift']
        votes={s:zb['level_fifty_'+s+'_up'] for s in ('random','ordered')}
    inputs={}
    parents={}
    hashes={}
    for key,source in [('parent',BASE),('fare',FARE),('partial',PARTIAL)]:
        file=ART/source/'test_probabilities.csv'
        p=pd.read_csv(file,float_precision='round_trip')
        manifest=json.loads((ROOT/'reports'/source/'manifest.json').read_text())
        path=ART/source/'submission.csv'
        parents[source]=dig(path)
        assert parents[source]==manifest['sha256']
        hashes[source]=dig(file)
        np.testing.assert_array_equal(p.PassengerId,ids[n:])
        np.testing.assert_array_equal(p.probability,p.reload_probability)
        np.testing.assert_array_equal(p.probability>=.5,pd.read_csv(path).Transported)
        inputs[key]=p.probability.to_numpy()
    np.testing.assert_array_equal(parent[n:],inputs['parent']>=.5)
    np.testing.assert_array_equal(missing[n:],(inputs['partial']>=.5)!=parent[n:])
    result={}
    for split,act in votes.items():
        np.testing.assert_array_equal(act[n:],(inputs['fare']>=.5)!=parent[n:])
        assert not (missing&act).any() and not (act&incomplete).any() and not (missing&~incomplete).any()
        total=missing|act
        diff=total[:n]*np.where(parent[:n]!=y,1,-1)
        result[split]=dict(test=int(total[n:].sum()),test_overlap=int((act[n:]&missing[n:]).sum()),
            scopes={s:count(total[:n],m,parent[:n]!=y) for s,m in [('all',known),('fresh',known&fresh),('blind',blind)]},
            fold_net=[int(diff[known&(f==j)].sum()) for j in range(8)])
        assert result[split]['scopes']['all']['fixed']>result[split]['scopes']['all']['harmed']
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    model=dest/'inputs.npz'
    inputs['incomplete']=incomplete[n:]
    probability=apply(**inputs)
    np.savez_compressed(model,passenger=ids[n:],**inputs)
    with np.load(model) as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:])
        repeat=apply(**{k:z[k] for k in inputs})
    np.testing.assert_array_equal(probability,repeat)
    changed=(probability>=.5)!=parent[n:]
    np.testing.assert_array_equal(changed,total[n:])
    assert changed.sum()==10
    csv=dest/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=probability>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=probability,reload_probability=repeat)).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(r.read_text())['file_sha256']!=check['sha256'] or r.parents[1].name==NAME
               for r in (ROOT/'reports').glob('*/online_attempts/*.json'))
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),training_rows=n,parents=parents,input_artifact_hashes=hashes,
                  reload_difference=0,model_sha256=dig(model),changed=int(changed.sum()),changed_orders=int(test.PassengerId[changed].str[:4].nunique()),
                  source_file='export_joint_masked_fare_n27.py')
    selection=dict(version=manifest['version'],rule='Use the frozen fare change on complete bills and the frozen two-bank mask change on naturally incomplete bills.',evidence=str(REPORT.relative_to(ROOT)),results=result)
    for folder in (dest,ROOT/'reports'/NAME):
        folder.mkdir(exist_ok=True);write(folder/'manifest.json',manifest);write(folder/'selection.json',selection)
    print(check['sha256'],result)


if __name__=='__main__':
    run()
