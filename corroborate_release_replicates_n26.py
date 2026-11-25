"""Separate recalibration changes that restore the public champion or are supported by outside families."""
import json
import numpy as np
from arbitration_against_scored_nine_n26 import champions
from full_roster_booking_validation_n23 import ROOT,data
from outside_panel_coverage_n26 import archives
from replicate_release_calibration_n26 import OUT
from solo_archive_recovery_n26 import ticker


def run():
    raw,y,g,fold=data()
    nine,known,_,_=champions()
    external,_=archives(raw,y,g,fold,True)
    scores=np.column_stack([np.mean([external[k] for k in ['Mitra1_frozen','Mitra2_frozen','Mitra2_adapted']],axis=0),
                            *[external[k] for k in ['TabDPT','LimiX2','EXAONE_grouped']]])
    report={}
    for tag in ('full','bookingmix','historical_to_new'):
        with np.load(OUT/f'decisions_{tag}.npz') as z:
            ids,parent,wrong=z['ids'],z['parent'],z['wrong']
            np.testing.assert_array_equal(parent!=y[ids],wrong)
            assert known[ids].all()
            groups={'champion_support':nine[ids]!=parent, 'champion_opposes':nine[ids]==parent,
                    'foundation_majority':((scores[ids]>=.5)!=parent[:,None]).sum(1)>=3}
            result={}
            for rule in ['one_opposes','majority_opposes','all_oppose','pooled_small_scores']:
                edits=z[rule]
                r={}
                for name,gate in groups.items():
                    select=edits&gate
                    by_fold=[int(np.where(wrong[m],1,-1).sum()) for k in np.unique(fold) if (m:=select&(fold[ids]==k)).any()]
                    r[name]=dict(**ticker(np.ones(len(ids),bool),wrong,select),active_fold_net=by_fold)
                result[rule]=r
            report[tag]=result
            print(tag,{n:{k:(v['fixed'],v['harmed']) for k,v in r.items()} for n,r in result.items()},flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/corroborate_release_replicates.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
