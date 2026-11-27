"""Apply distinct error evidence to three mutually exclusive payment states on the strongest public parent."""
import argparse
import json
import numpy as np
import pandas as pd
from export_consensus_review_n26 import ROOT, ART, write, dig
from full_roster_booking_validation_n23 import data
from kaggle_workflow import validate_submission
from repeated_archive_misses_n26 import cohort,count
from price_implied_deck_n26 import FEES
from six_way_against_scored_nine_n26 import test_decisions

REPORT=ROOT/'reports/nov23_matched_learning_curve/route_by_bill_observation.json'
NAME='proposal_n27_three_payment_states'
BASE='proposal_n27_directional_foundation'


def run(export):
    raw,y,g,f=data()
    _,blind,known,fresh=cohort(raw,y,f)
    frame=pd.concat([raw,pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns)],ignore_index=True)
    n=len(y);ids=frame.PassengerId.to_numpy(str)
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        first=z['first']
        lock=z['random_median']
        current=first^lock
    with np.load(ART/'nov27_partial_receipt_deck/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        missing=z['missing']
        incomplete=missing & ((z['base']>=.5)==current) & ((z['augmented']>=.5)!=current)
    with np.load(ART/'nov27_sleeper_companion_tabicl/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        np.testing.assert_array_equal(z['anchor'],current)
        sleeper=z['rule_real_change_both']  # Require the matched peer classifier itself to oppose the parent.
    paid=frame[FEES].gt(0).any(axis=1).to_numpy()
    sleeper=sleeper&~paid&~missing
    directory=ART/'nov27_adjudicate_paid_release_paths'
    with np.load(directory/'test_full.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:])
        np.testing.assert_array_equal(z['base'],first[n:])
        test_actions={k:z[k] for k in ['real','paths','control']}
    report={}; test_flags=None; align={}
    for model in test_actions:
        for scheme in ['random','ordered']:
            with np.load(directory/f'full_{scheme}.npz') as z:
                bill=np.r_[z[model],test_actions[model]] & paid & ~missing & ~lock
            sets=np.column_stack([incomplete,sleeper,bill])
            assert sets.sum(1).max()==1
            rule=sets.any(1)
            err=current[:n]!=y
            def report_one(a):
                delta=a[:n]*np.where(err,1,-1)
                return dict(all=count(a[:n],known,err),fresh=count(a[:n],known&fresh,err),old=count(a[:n],blind,err),
                    folds=[int(delta[known&(f==k)].sum()) for k in range(8)],test=int(a[n:].sum()))
            report.setdefault(model,{})[scheme]={k:report_one(v) for k,v in zip(['partial','zero','paid','union'],[*sets.T,rule])}
            if model=='real':
                align[scheme]=sets
                if test_flags is None:test_flags=sets[n:]
                else:np.testing.assert_array_equal(sets[n:],test_flags)
    cache=ART/'nov27_payment_state_routes'
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache/'aligned.npz',passenger=ids,y=y,fold=f,parent=current,**align)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    for k,r in report.items():print(k,r['random']['union'],r['ordered']['union'])
    print('parts',report['real']['random'])
    if not export:return
    dest=ART/NAME
    dest.mkdir(exist_ok=True)
    hard,sha=test_decisions(BASE,pd.Series(ids[n:]))
    par=pd.read_csv(ART/BASE/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(ids[n:],par.PassengerId)
    np.testing.assert_array_equal(par.probability>=.5,hard)
    np.testing.assert_array_equal(hard,current[n:])
    source=dest/'inputs.npz'
    np.savez_compressed(source,passenger=ids[n:],parent=par.probability.to_numpy(),partial=test_flags[:,0],
                        zero=test_flags[:,1],paid=test_flags[:,2])
    # Store decision scores with the parent's original distance from .5 on any reversed decision.
    def apply(parent,changes):
        return np.where(changes,np.where(parent==.5,np.nextafter(.5,0.),1.-parent),parent)
    p=apply(par.probability.to_numpy(),test_flags.any(1))
    with np.load(source) as z:
        replay=apply(z['parent'],z['partial']|z['zero']|z['paid'])
    np.testing.assert_array_equal(replay,p)
    np.testing.assert_array_equal(p>=.5,hard^test_flags.any(1))
    csv=dest/'submission.csv'
    pd.DataFrame({'PassengerId':ids[n:],'Transported':p>=.5}).to_csv(csv,index=False)
    pd.DataFrame({'PassengerId':ids[n:],'probability':p,'reload_probability':replay}).to_csv(dest/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    assert all(json.loads(a.read_text())['file_sha256']!=check['sha256'] for a in (ROOT/'reports').glob('*/online_attempts/*.json'))
    provenance=[directory/'test_full.npz',ART/'nov27_partial_receipt_deck/aligned.npz',ART/'nov27_sleeper_companion_tabicl/aligned.npz']
    manifest=dict(**check,version=NAME.removeprefix('proposal_'),parents={BASE:sha},training_rows=n,
                  input_artifact_hashes={str(p.relative_to(ART)):dig(p) for p in provenance},
                  model_sha256=dig(source),reload_difference=0,changed=int(test_flags.any(1).sum()),source_file='route_by_bill_observation_n27.py')
    for folder in (dest,ROOT/'reports'/NAME):
        write(folder/'manifest.json',manifest)
        write(folder/'selection.json',dict(version=manifest['version'],policy='Partial bills use mask augmentation, complete zero bills use matched companions, complete positive bills use the full-order four-source reviewer.',evidence=str(REPORT.relative_to(ROOT))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true')
    run(parser.parse_args().export)
