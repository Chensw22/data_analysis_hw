"""Pair the frozen 18-change release with the actually scored nine-vote release."""
import json
from hashlib import sha256
import numpy as np
import pandas as pd
from outside_panel_coverage_n26 import ROOT, projection
from family_identity_transfer_n26 import panels_and_reference
from release_booking_composition_n26 import standardize


def read(path):
    return json.loads((ROOT/path).read_text())


def test_decisions(key, ids):
    path = ROOT/'artifacts'/key/'submission.csv'
    digest = sha256(path.read_bytes()).hexdigest()
    assert digest == read(f'reports/{key}/manifest.json')['sha256']
    frame = pd.read_csv(path)
    np.testing.assert_array_equal(frame.PassengerId, ids)
    assert frame.Transported.dtype==bool
    return frame.Transported.to_numpy(),digest


def transfer_edits(before, after, truth, groups, target_up, target_down, test_rows):
    # Bootstrap whole held bookings; a booking can contain both change directions.
    _, booking = np.unique(groups,return_inverse=True)
    columns = []
    for active in ((~before)&after, before&(~after)):
        signed = active*(2*(after==truth).astype(float)-1)
        columns.extend((np.bincount(booking,weights=active),np.bincount(booking,weights=signed)))
    table = np.array(columns).T
    def estimate(t):
        return 100*(target_up*t[...,1]/t[...,0]+target_down*t[...,3]/t[...,2])/test_rows
    generator = np.random.default_rng(131)
    values = []
    for _ in range(80):
        drawn = generator.integers(len(table),size=(25,len(table)))
        stat = table[drawn].sum(1)
        valid=(stat[:,0]>0)&(stat[:,2]>0)
        values.extend(estimate(stat[valid]))
    return dict(point_gain_pp=float(estimate(table.sum(0))), cluster_95_pp=np.quantile(values,[.025,.975]).tolist(),
                valid_bootstrap_draws=len(values))


def run():
    best = read('reports/current_best_submission.json')
    receipt = read(best['online_record'])
    assert receipt['evaluation']['Status']=='COMPLETE'
    assert receipt['submission_ref']==best['submission_ref']==receipt['evaluation']['Submission Ref']
    assert float(receipt['evaluation']['Public Score'])==best['public_score']
    assert best['proposal']=='proposal_n24_selected_dissent'
    test = pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    tgroup = test.PassengerId.str.split('_').str[0]
    champion,hd = test_decisions(best['proposal'],test.PassengerId)
    candidate,cd = test_decisions('proposal_n26_six_way_dispute',test.PassengerId)
    assert hd==receipt['file_sha256']==best['csv_sha256']
    tsize = tgroup.groupby(tgroup).transform('size').to_numpy()
    report = dict(official_receipt=best['online_record'],official_ref=best['submission_ref'],public_score=best['public_score'],
                  test=dict(champion_sha256=hd,candidate_sha256=cd,rows=len(candidate),
                            to_positive=int((candidate&~champion).sum()),to_negative=int((~candidate&champion).sum())),
                  local={})
    raw,y,g,f,views,_ = projection(include_full=True)
    _,_,fold,_,_,_,_,_,_,reference=panels_and_reference()
    np.testing.assert_array_equal(f,fold)
    with np.load(ROOT/'artifacts/nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
        np.testing.assert_array_equal(z['row'],np.arange(len(y)))
        np.testing.assert_array_equal(z['fold'],f)
        np.testing.assert_array_equal(z['original_prior'],reference[0])
        prior=z['prior']
    known = prior>=0
    np.testing.assert_array_equal(known,views['full'][0])
    nine = np.where(prior==1,reference[2],reference[1]).astype(bool)
    size = raw.PassengerId.groupby(g).transform('size').to_numpy()
    for tag in ('full','bookingmix','historical_to_new'):
        mask,child=views[tag]
        assert not (mask&~known).any()
        ids=np.flatnonzero(mask)
        dr=(child==y).astype(int)-(nine==y)
        net=int(dr[mask].sum())
        unknown=len(y)-len(ids)
        r=dict(rows=len(ids),nine_correct=int((nine[mask]==y[mask]).sum()),candidate_correct=int((child[mask]==y[mask]).sum()),
               fixed=int((dr[mask]>0).sum()),harmed=int((dr[mask]<0).sum()),net=net,
               fold_net=[int(dr[mask&(f==k)].sum()) for k in np.unique(f)],
               false_negatives_fixed=int((mask&(dr>0)&y).sum()),false_positives_fixed=int((mask&(dr>0)&~y).sum()),
               false_negatives_added=int((mask&(dr<0)&y).sum()),false_positives_added=int((mask&(dr<0)&~y).sum()))
        for cap in (4,8):
            r[f'test_size_cap_{cap}']=standardize(ids,size,tsize,g,y,nine,child,cap)
        r['conditional_on_test_edit_counts']=transfer_edits(nine[mask],child[mask],y[mask],g[mask],
                report['test']['to_positive'],report['test']['to_negative'],len(test))
        if tag=='full':
            r['unpaired']=unknown
            r['whole_roster_net_bounds']=[net-unknown,net+unknown]
        report['local'][tag]=r
        print(tag,r,flush=True)
    (ROOT/'reports/nov23_matched_learning_curve/six_way_against_scored_nine.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['test'])


if __name__=='__main__':
    run()
