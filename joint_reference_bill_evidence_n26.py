"""Combine booking-held differences specific to reference people, bill interactions and real receipts."""
import json
import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

from full_roster_booking_validation_n23 import ROOT, data
from repeated_archive_misses_n26 import ART, cohort, count

OUT=ART/'nov26_joint_reference_bill_evidence'
REPORT=ROOT/'reports/nov23_matched_learning_curve/joint_reference_bill_evidence.json'


def odds(p):
    return logit(p.clip(.001,.999))


def run():
    raw,y,g,f=data()
    base,blind,known,fresh=cohort(raw,y,f)
    n=len(y)
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        ids,anchor=z['passenger'],z['anchor']
        np.testing.assert_array_equal(ids[:n],raw.PassengerId)
        np.testing.assert_array_equal(anchor[:n],base)
        native=np.where(anchor,1.,-1.)*abs(odds(z['scores'][:,-3:].mean(1)))
    def read(path,id_key,keys):
        with np.load(ART/path) as z:
            for k,expected in [(id_key,ids),('y',y),('fold',f)]:
                np.testing.assert_array_equal(z[k],expected)
            return [z[k] for k in keys]
    identity,anonymous,reassigned=read('nov26_passenger_reference_landmarks/aligned.npz','passenger',
                                        ['identity','anonymous','reassigned'])
    named,magnitude,main=read('nov26_bill_mask_harmonics/scores.npz','passenger',['named','magnitude','main'])
    real,sham,direct=read('nov26_paired_manifest_density_ratio/aligned.npz','ids',['true_log_ratio','sham_log_ratio','direct'])
    # A matched difference for each source. Keep the score common to all three sources exactly once.
    pair=odds(identity)-odds(anonymous)
    mask=odds(named)-odds(magnitude)
    receipt=real-sham
    evidence=np.column_stack([pair,mask,receipt])
    correction=evidence*np.where(anchor,-1.,1.)[:,None]
    crossed=(native[:,None]+evidence>=0)!=anchor[:,None]
    comparable=np.column_stack([odds(anonymous)-odds(reassigned),odds(magnitude)-odds(main),sham])
    controls=(native[:,None]+comparable>=0)!=anchor[:,None]
    pairs=crossed[:,0]&crossed[:,1]
    flags=dict(pair_and_mask=pairs,pair_and_receipt=crossed[:,0]&crossed[:,2],mask_and_receipt=crossed[:,1]&crossed[:,2],
               majority=crossed.sum(1)>=2,controls_majority=controls.sum(1)>=2,
               unanimous=crossed.all(1),sum=((native+evidence.sum(1))>=0)!=anchor,
               pair_mask_sum=((native+pair+mask)>=0)!=anchor,
               conditional_receipt=pairs&(correction[:,2]>0),direct_confirmed=pairs&((direct>=.5)!=anchor))
    with np.load(ART/'nov26_repeated_archive_misses/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[:n])
        names=z['names']
        subset=np.array([name.startswith(('nov24_passenger_pair_outcomes/','nov24_passenger_pair_outcomes_repeat/','nov24_bill_subset_route/')) for name in names])
        covered=(((z['full'][:,subset]-.5)*np.where(base,-1.,1.)[:,None])>0).any(1)
    assert int((covered&blind&(base!=y)).sum())==158
    scopes=dict(all=known,fresh=known&fresh,common=blind,original_union=blind&covered)
    result={}
    for k,flag in flags.items():
        delta=flag[:n]*np.where(base!=y,1,-1)
        result[k]=dict(scopes={s:count(flag[:n],v,base!=y) for s,v in scopes.items()},
                       test_changes=int(flag[n:].sum()),by_fold=[int(delta[known&(f==i)].sum()) for i in range(8)])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'actions.npz',passenger=ids,y=y,fold=f,anchor=anchor,**flags)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print([(k,v['scopes']['all'],v['scopes']['original_union']) for k,v in result.items()])


if __name__=='__main__':
    with threadpool_limits(1):
        run()
