"""Retrospective selection audit across scored files with matched OOF decisions and verified submissions."""
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_recent_family_transfer_n23 import RELEASES as PRIOR_FAMILIES
from backtest_frozen_queue_forecast_n26 import BEST, NAMES as N25
from broad_geographic_transfer_n23 import load
from broad_observed_state_transfer_n23 import decide as state_decision
from broad_partition_evidence_n23 import ART, ROOT
from broad_release_transfer_n23 import matched
from broad_three_corrections_replay_n23 import scores
from export_consensus_review_n26 import dig
from full_roster_edit_resampling_n24 import calibrate, features, shift, subsample
from pending_family_coverage_n23 import PAIRS,development_votes,rules
from release_vote_information_n23 import DIRS
from released_correction_audit_n23 import development

REPORT=ROOT/'reports/nov23_matched_learning_curve/cross_family_submission_choices.json'
V131='proposal_v131_party_consensus'


def read(path):
    return json.loads(path.read_text())


def inventory(ids):
    out={}
    for path in sorted((ROOT/'reports').glob('*/online_attempts/*.json')):
        item=read(path)
        e=item.get('evaluation',{})
        if e.get('Status')!='COMPLETE':
            continue
        key=item['run']
        file=ART/key/'submission.csv'
        pred=pd.read_csv(file)
        assert item['competition']=='spaceship-titanic' and e['Submission Ref']==str(item['submission_ref'])
        assert dig(file)==item['file_sha256'] and pred.Transported.dtype==bool and key not in out
        np.testing.assert_array_equal(pred.PassengerId,ids)
        assert item['rows']==len(ids)==4277
        out[key]=dict(score=float(e['Public Score']),ref=e['Submission Ref'],sha256=item['file_sha256'],
                      date=e['Submission Date'][:10],submitted=e['Submission Date'],receipt=str(path.relative_to(ROOT)),
                      vote=pred.Transported.to_numpy())
    return out


def paired(v,b,truth):
    assert len(v)==len(b)==len(truth)
    return [int(((v!=b)&(v==truth)).sum()),int(((v!=b)&(v!=truth)).sum())]


def bagged_historical(raw,row,y,group,fold,native,ref):
    # Replay only the historical branch; the production training routine also writes its source report.
    size=raw.groupby(raw.PassengerId.str.split('_').str[0]).PassengerId.transform('size').to_numpy()
    x=features(raw.iloc[row],size[row],native)
    p=native[:,0]
    estimates=[]
    for seed in (0,1443,1444,1445):
        shifts=np.empty(len(row))
        for k in np.unique(fold):
            ids=np.flatnonzero(fold==k)
            for a,b in GroupKFold(4,shuffle=True,random_state=633).split(ids,groups=group[ids]):
                t,q=ids[a],ids[b]
                t=t[(p[t]>=.5)==ref[t]]
                assert not np.isin(group[t],group[q]).any()
                models=[calibrate(x,p,y,group,subsample(t,group,seed+d if seed else 0)) for d in (0,57)]
                shifts[q]=np.mean([shift(m,x[q]) for m in models],axis=0)
        estimates.append(shifts)
    predicted=np.where((p>=.5)==ref,logit(p.clip(.003,.997))+np.mean(estimates,axis=0)>=0,ref)
    expected=read(ROOT/'reports/nov23_matched_learning_curve/edit_resampling.json')['bagged']['historical']
    assert paired(predicted,ref,y)==[expected['corrected'],expected['harmed']]
    return predicted


def recent(inv):
    row,y,frame,g,_=load()
    w=read(PAIRS/'test_budget.json')['test']['change_150']['weight']
    ref,votes=development_votes(frame,g,row,y,'v131',w)
    base,r=scores(frame,g,row,y,'v131')
    np.testing.assert_array_equal(r,ref)
    v={k:p for k,p in votes.items() if k in inv}
    v.update(proposal_n23_partition_residual=base[:,1]>=.5,proposal_n23_posterior_spending_risk=base[:,2]>=.5,
             proposal_n24_two_family_support=ref^rules(ref,votes)[0]['two_families'])
    with np.load(ART/'nov23_joint_observed_state/v131.npz') as z:
        for label,p in [('row',row),('y',y),('ref',ref)]:np.testing.assert_array_equal(z[label],p)
        for name,view,which in [('direct_basket_quarter','basket','direct'),('joint_basket_quarter','basket','joint'),
                                ('sleep_basket_joint_quarter','mean','joint')]:
            v['proposal_n23_'+name]=state_decision(z['base'],ref,z[view+'_conditional'],z[view+'_direct'])['quarter_'+which]
    with np.load(ART/'nov23_release_vote_information/panel.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        np.testing.assert_array_equal(z['y'],y)
        paper=z['votes']
        assert paper.shape==(2,len(row),5)
        np.testing.assert_array_equal(z['official'],np.column_stack([inv[n]['vote'] for n in DIRS]))
        for j,n in enumerate(DIRS):
            if j!=3:
                np.testing.assert_array_equal(paper[0,:,j],paper[1,:,j])
                v[n]=paper[0,:,j]
        np.testing.assert_array_equal(ref,v.pop(V131))
        variant={DIRS[3]:paper[:,:,3],'proposal_n23_five_release_majority':paper.sum(2)>=3}
        np.testing.assert_array_equal(inv['proposal_n23_five_release_majority']['vote'],z['official'].sum(1)>=3)
    for n,(_,path,key) in PRIOR_FAMILIES.items():
        expected=read(ROOT/'reports'/path)['plans']['v131'][key]
        assert paired(v[n],ref,y)==[expected['fixed'],expected['harmed']]
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],row)
        np.testing.assert_array_equal(z['y'],y)
        np.testing.assert_array_equal(z['v131'],ref)
        v[BEST]=z['nine']
        p=z['scores'][:,4]
        hf=z['fold']
    with np.load(ART/'nov24_remaining_cases/archive_scores.npz') as z:
        np.testing.assert_array_equal(z['row'],row)
        challenger=z['historical'][:,z['names'].tolist().index('permutation_selection/random')]
    random='proposal_n24_random_subspace'
    v[random]=np.where((p>=.5)==ref,.75*p+.25*challenger>=.5,ref)
    solo=frame.BookingSize.eq(1).to_numpy()
    v['proposal_n24_solo_random_nine']=np.where(v[BEST]!=ref,v[BEST],np.where(solo,v[random],ref))
    raw=pd.read_csv(ROOT/'data/raw/train.csv')
    from full_roster_champion_overlap_n24 import aligned
    _,ty,_,ids,fold,_,rb,_,_,_,_,old,_=aligned()
    for left,right in [(row,ids),(ty,y),(rb,ref),(fold,hf)]:np.testing.assert_array_equal(left,right)
    v['proposal_n24_bagged_net_edits']=bagged_historical(raw,row,y,g,hf,old,ref)
    return ref,v,y,variant


def summary(options):
    rows=list(options.values())
    def winners(key):
        best=max(v[key] for v in rows)
        return [v for v in rows if v[key]==best]
    top=winners('local_net')
    high=winners('public_gain_pp')
    ordered={'more_than_one_pp':[],'within_one_pp':[]}
    for a,b in combinations(rows,2):
        loc=a['local_net']-b['local_net']
        pub=round(a['public_gain_pp']-b['public_gain_pp'],8)
        group='more_than_one_pp' if abs(loc)*100/a['local_rows']>1 else 'within_one_pp'
        ordered[group].append(int(np.sign(loc)*np.sign(pub)))
    return dict(candidates=len(rows),local_winners=[v['proposal'] for v in top],
                public_winners=[v['proposal'] for v in high],
                local_winner_is_public_best=any(v in high for v in top),
                chosen_public_gain_range_pp=[min(v['public_gain_pp'] for v in top),max(v['public_gain_pp'] for v in top)],
                chosen_local_gain_pp=top[0]['local_net']*100/top[0]['local_rows'],best_public_gain_pp=high[0]['public_gain_pp'],
                local_positives=sum(v['local_net']>0 for v in rows),
                local_positives_public_positive=sum(v['local_net']>0 and v['public_gain_pp']>0 for v in rows),
                local_positives_public_tied=sum(v['local_net']>0 and v['public_gain_pp']==0 for v in rows),
                pairwise={k:{'concordant':v.count(1),'opposite':v.count(-1),'tied':v.count(0)} for k,v in ordered.items()})


def mid_november(row, y, versions, inv, add):
    from champion_tree_blend import saved_oof
    from relational_error_corrector import logit as release_logit

    raw=pd.read_csv(ROOT/'data/raw/train.csv').iloc[row]
    folders={106:'proposal_nov18_peer_cabin',108:'proposal_v108_party_bills',
             109:'proposal_v109_champion_tree',110:'proposal_v110_party_counts',
             113:'proposal_v113_population_transfer'}
    with np.load(ART/'v108_exact_oof/oof.npz') as exact:
        for key,value in [('dev_ids',row),('PassengerId',raw.PassengerId),('true',y)]:
            np.testing.assert_array_equal(exact[key],value)
        p106,p108,fold=(exact[k] for k in ('v106','v108','fold_id'))
    np.testing.assert_array_equal(versions[108],p108>=.5)
    with saved_oof('teacher_distillation') as z:
        for k,value in [('dev_ids',row),('passenger_ids',raw.PassengerId),('fold_id',fold)]:
            np.testing.assert_array_equal(z[k],value)
        p109=(p106+z['hard'])/2
    with np.load(ART/'party_outcome_count.npz') as z:
        np.testing.assert_array_equal(row,z['dev'])
        p110=expit(release_logit(p106)+z['shift'])
    with np.load(ART/'population_shift.npz') as z:
        np.testing.assert_array_equal(row,z['dev'])
        p113=expit(release_logit(p108)+.5*(release_logit(z['target_weighted'])-release_logit(z['native'])))
    for child, anchor, prediction, reference in [(108,106,p108,p106),(109,106,p109,p106),
                                                   (110,106,p110,p106),(113,108,p113,p108)]:
        n,b=folders[child],folders[anchor]
        manifest=read(ROOT/'reports'/n/'manifest.json')
        assert manifest['source_csv']==inv[b]['sha256']
        changed=int((inv[n]['vote']!=inv[b]['vote']).sum())
        assert changed==manifest.get('decisions_changed',manifest.get(f'changed_vs_v{anchor}'))
        add('released_correction',n,b,len(y),paired(prediction>=.5,reference>=.5,y),
            'Exact v106/v108 historical probabilities plus the released child transformation and authenticated parent test SHA.')


def aggregate(choices):
    choice=[v for v in choices.values() if v['candidates']>=2]
    signed=[r for b in choices.values() for r in b['versions'].values() if r['local_net']>0]
    def pairs(key):
        return {s:sum(b['pairwise'][key][s] for b in choice) for s in ['concordant','opposite','tied']}
    return dict(distinct_primary_candidates=len({r['proposal'] for b in choices.values() for r in b['versions'].values()}),
        two_or_more_choice_sets=len(choice),public_best_among_local_top=sum(v['local_winner_is_public_best'] for v in choice),
        random_selection_expected_public_winners=sum(len(v['public_winners'])/v['candidates'] for v in choice),
        local_winner_beats_anchor_for_all_ties=sum(v['chosen_public_gain_range_pp'][0]>0 for v in choice),
        local_winner_always_under_anchor=sum(v['chosen_public_gain_range_pp'][1]<0 for v in choice),
        mean_public_regret_pp=float(np.mean([b['best_public_gain_pp']-np.mean(b['chosen_public_gain_range_pp']) for b in choice])) if choice else None,
        choice_set_sizes=[v['candidates'] for v in choice],any_candidate_above_anchor=sum(v['best_public_gain_pp']>0 for v in choice),
        local_positive_versions=len(signed),online_above=sum(r['public_gain_pp']>0 for r in signed),
        online_ties=sum(r['public_gain_pp']==0 for r in signed),online_below=sum(r['public_gain_pp']<0 for r in signed),
        within_one_pp=pairs('within_one_pp'),over_one_pp=pairs('more_than_one_pp'))


def run():
    ids=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId']).PassengerId
    inv=inventory(ids)
    by_ref={r['ref']:k for k,r in inv.items()}
    assert len(by_ref)==len(inv)
    batches=defaultdict(dict)
    covered=set()
    def add(stage,n,base,tot,counts,evidence):
        a,b=inv[n],inv[base]
        key='/'.join([stage,a['date'],base])
        fixed,harmed=map(int,counts)
        assert n not in batches[key] and base!=n
        batches[key][n]=dict(proposal=n,baseline=base,local_rows=tot,fixed=fixed,harmed=harmed,local_net=fixed-harmed,
                             public_gain_pp=round(100*(a['score']-b['score']),7),
                             test_edits=int((a['vote']!=b['vote']).sum()),evidence=evidence)
        covered.update([n,base])
    row,y,early=matched()
    for v in early:
        v['name']=by_ref[str(v['ref'])]
        np.testing.assert_array_equal(v['test'],inv[v['name']]['vote'])
    chronological=sorted(early,key=lambda v:v['time'])
    for date in sorted({v['time'][:10] for v in early}):
        previous=[v for v in chronological if v['time'][:10]<date]
        today=[v for v in chronological if v['time'][:10]==date]
        anchor=max(previous,key=lambda v:v['score']) if previous else today[0]
        for v in today:
            if v is not anchor:
                add('early_common_folds',v['name'],anchor['name'],len(y),paired(v['prob']>=.5,anchor['prob']>=.5,y),
                    dict(local_file=v['cached'],identity=v['id_check'],baseline_identity=anchor['id_check']))
    yy,versions,_=development()
    np.testing.assert_array_equal(yy,y)
    mid_november(row,y,versions,inv,add)
    old={i:next(n for n in inv if n.startswith(f'proposal_v{i}_')) for i in versions}
    for i in range(114,137):
        a=108 if i<129 else 128
        add('released_correction',old[i],old[a],len(yy),paired(versions[i],versions[a],yy),'released_correction_audit_n23.development')
    ref,v,ny,variant=recent(inv)
    np.testing.assert_array_equal(y,ny)
    np.testing.assert_array_equal(versions[131],ref)
    for n,p in v.items():
        add('v131_releases',n,V131,len(y),paired(p,ref,y),'historical v131 individual decisions')
    alt={}
    for n,p in variant.items():
        alt[n]=dict(anchor=V131,replays=[paired(x,ref,y) for x in p],
                    public_gain_pp=round(100*(inv[n]['score']-inv[V131]['score']),7),
                    different_passengers=int((p[0]!=p[1]).sum()))
    n25=read(REPORT.with_name('backtest_frozen_queue_forecast.json'))
    assert n25['reference']['sha256']==inv[BEST]['sha256']
    for n in N25:
        r=n25['scored_versions'][n]
        assert r['receipt']['sha256']==inv[n]['sha256'] and r['receipt']['ref']==inv[n]['ref']
        for tag in ['historical','eight_known']:
            if tag in r:
                q=r[tag]
                add('n25_'+tag,n,BEST,q['rows'],[sum(q[k]) for k in ['down_up_fixed','down_up_harmed']],
                    'backtest_frozen_queue_forecast_n26.py')
    results={k:dict(**summary(v),versions=v) for k,v in batches.items()}
    strict={k:v for k,v in results.items() if not k.startswith('n25_eight_known')}
    summary_report=aggregate(strict)
    family_breakdown={stage:aggregate({k:v for k,v in strict.items() if k.split('/')[0]==stage})
                      for stage in sorted({k.split('/')[0] for k in strict})}
    explicit_only={}
    for key,batch in strict.items():
        opts=batch['versions']
        if key.startswith('early_common_folds'):
            opts={n:r for n,r in opts.items() if r['evidence']['identity']==r['evidence']['baseline_identity']=='row IDs'}
        if opts:
            explicit_only[key]=dict(**summary(opts),versions=opts)
    public={k:{n:r[n] for n in ['score','ref','date','sha256','receipt']} for k,r in inv.items()}
    out=dict(inventory=public,verified_receipts=len(inv),with_primary_local_evidence=len(covered),
        summary=summary_report,by_source=family_breakdown,excluding_early_implicit_ids=aggregate(explicit_only),batches=results,variant_calibration_only=alt,
        outside_primary=sorted(set(inv)-covered),
        design='Within each UTC date, common historical replay population, and common reference, select the largest paired local net among submitted alternatives. Ties are reported. Early baselines use the best previously scored member of the same early panel, or the first evaluated on the first day. Later blocks use the actual historical anchor. Single-candidate groups do not count as choices. Eightfold replays of the same four N25 files are separate, not counted twice. No other public results are used to construct local predictions.',
        limitations='Only submitted files are observable; dates are retrospective comparison groups, not proof that every file was available simultaneously. Candidate versions, dates, and public data overlap, so counts are descriptive. Early model files lacking row ids were matched by recorded split accuracies and loss; weaker than explicit id checks. Variant calibration-only entries are not ranked.')
    REPORT.write_text(json.dumps(out,indent=2)+'\n')
    print(summary_report)
    print('coverage',len(covered),len(inv),len(alt))
    for k,v in strict.items():
        print(k,v['candidates'],v['local_winners'],v['chosen_public_gain_range_pp'],v['best_public_gain_pp'])


if __name__=='__main__':
    with threadpool_limits(1):run()
