"""Add independent profile-location and receipt reviews to the scored 42-passenger baseline."""
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from archive_source_harm_veto_n28 import inputs
from export_budget_quarter_n23 import source
from foundation_support_for_mechanisms_n28 import decisions
from kaggle_workflow import validate_submission,sha256
from repeated_archive_misses_n26 import ART,ROOT,count,cohort
from relational_error_corrector import write

BASE='proposal_n28_tied_raw_crosscheck'
DIRECTION='proposal_n27_directional_foundation'
NAME='proposal_n28_profile_and_bill_review'


def flags(direction,tied,seven,raw,absolute,eligible,views):
    strong=(raw.any(1))&(((seven[:,1:4]>=.5)!=direction[:,None]).all(1))
    foundations=((seven[:,1:4]>=.5)!=direction[:,None]).sum(1)
    support=(views>=0)!=direction[:,None]
    domains=(support[:,:2].all(1)).astype(int)+support[:,2]+support[:,3]
    location=eligible&((absolute>=0)!=direction)&(foundations>=2)&(domains>=2)
    # Add new changes only where all previously tied versions predict the same outcome.
    common=(tied==direction[:,None]).all(1)
    return common&(strong|location)


def load_data(folder, ids, y, fold, tag):
    with np.load(ART/folder/'aligned.npz') as z:
        for k,v in [('ids',ids),('y',y),('fold',fold)]:np.testing.assert_array_equal(z[k],v)
        return z[tag]


def apply(z):
    edit=flags(z['direction'],z['tied'],z['seven'],z['raw'],z['absolute'],z['eligible'],z['views'])
    soft=z['soft']
    out=np.where(edit,1.-soft,soft);out[edit&(soft==.5)]=np.nextafter(.5,0)
    return out


def run():
    frame,y,g,fold,ids,_,_,seven,versions=inputs();n=len(y)
    direction=versions['random'][:,0]
    known,fresh=cohort(frame.iloc[:n],y,fold)[2:]
    soft,official,digest=source(BASE,frame.iloc[n:]);selection={}
    for tag,tied in versions.items():
        raw=load_data('nov28_foundation_support_for_mechanisms',ids,y,fold,tag)
        views=load_data('nov28_pooled_foundation_challenge',ids,y,fold,tag)
        loc=load_data('nov28_exact_profile_location_rank',ids,y,fold,tag)
        parent=direction^decisions(direction,tied,raw,seven)['any_tied_and_raw']
        np.testing.assert_array_equal(parent[n:],official)
        common=(tied==direction[:,None]).all(1)&(parent==direction)
        edit=flags(direction,tied,seven,raw,loc[:,0],loc[:,2]>0,views)
        assert not (edit&~common).any()
        hard=parent^edit
        row={}
        for name,baseline in [('direction',direction),('42',parent)]:
            wrong=baseline[:n]!=y;change=hard[:n]!=baseline[:n]
            scopes={'all':known,'new_orders':known&fresh,'five_tied':known&common[:n],
                'five_and_seven':known&common[:n]&((seven[:n]>=.5)==direction[:n,None]).all(1)}
            row[name]={s:count(change,m,wrong) for s,m in scopes.items()}
            row[name]['test']=int((hard[n:]!=baseline[n:]).sum())
        selection[tag]=row
        if tag=='random':
            packet=dict(ids=ids[n:],soft=soft,direction=direction[n:],tied=tied[n:],raw=raw[n:],
                seven=seven[n:],views=views[n:],absolute=loc[n:,0],eligible=loc[n:,2]>0)
            test_hard=hard[n:]
        else: np.testing.assert_array_equal(test_hard,hard[n:])
    print(selection,flush=True)
    write(ROOT/'reports/nov23_matched_learning_curve/profile_and_bill_review.json',selection)
    if not all(row['42']['all']['fixed']>row['42']['all']['harmed'] for row in selection.values()):return
    out=ART/NAME;out.mkdir(exist_ok=True)
    model=out/'inputs.npz';np.savez_compressed(model,**packet)
    p=apply(packet)
    with np.load(model) as z:reloaded=apply(z)
    np.testing.assert_array_equal(p,reloaded);np.testing.assert_array_equal(p>=.5,test_hard)
    csv=out/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=p>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=p,reload_probability=reloaded)).to_csv(out/'test_probabilities.csv',index=False)
    checked=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert not any(json.loads(v.read_text()).get('file_sha256')==checked['sha256'] for v in (ROOT/'reports').glob('*/online_attempts/*.json'))
    version=NAME.removeprefix('proposal_')
    for d in [out,ROOT/'reports'/NAME]:
        write(d/'manifest.json',dict(version=version,**checked,model_sha256=sha256(model),parents={BASE:digest},training_rows=n,
                                      changed=int((test_hard!=official).sum()),reload_difference=0))
        write(d/'selection.json',dict(version=version,reference=BASE,evidence=selection,
              rule='Outside disagreements of the five tied submissions, accept corroborated exact-profile location estimates or unanimous foundation opposition with a raw-data increment.',
              limitation='These cohorts and methods were explored on the same labeled bookings.'))


if __name__=='__main__':
    with threadpool_limits(1):run()
