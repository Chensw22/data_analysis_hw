"""Audit locally validated branches against both scored parents and official prediction files."""
import hashlib
import io
import json

import numpy as np
import pandas as pd

from directional_foundation_transfer_n27 import calibrate
from dissent_likelihood_transfer_n27 import splits
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort,count
from six_way_against_scored_nine_n26 import test_decisions

OUT=ART/'nov27_unsubmitted_audit'
REPORT=ROOT/'reports/nov23_matched_learning_curve/unsubmitted_branch_audit.json'
FIRST='proposal_n26_champion_fare_six'
BEST='proposal_n27_directional_foundation'


def digest(ids,pred):
    stream=io.StringIO()
    pd.DataFrame(dict(PassengerId=ids,Transported=pred)).to_csv(stream,index=False)
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


def run():
    raw,y,g,f=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv')
    ids=np.r_[raw.PassengerId.to_numpy(str),test.PassengerId.to_numpy(str)]
    _,blind,known,fresh=cohort(raw,y,f)
    d=ART/'nov27_dissent_likelihood_transfer'
    with np.load(d/'aligned.npz') as z:
        np.testing.assert_array_equal(ids,z['passenger'])
        first=z['first']
        locks=[z['random_median'],z['contiguous_median']]
    best=[first^m for m in locks]
    for k,v in [(FIRST,first),(BEST,best[0])]:
        side,sha=test_decisions(k,test.PassengerId)
        np.testing.assert_array_equal(side,v[n:])
        assert digest(ids[n:],side)==sha
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:
        fixed=z['parent']
        np.testing.assert_array_equal(best[0],fixed)
    receipts={}
    by_vector={}
    for p in (ROOT/'reports').glob('*/online_attempts/*.json'):
        logged=json.loads(p.read_text())
        key=logged.get('file_sha256')
        if not key:continue
        receipts.setdefault(key,[]).append(p.parents[1].name)
        csv=ART/p.parents[1].name/'submission.csv'
        if csv.is_file():
            hist=pd.read_csv(csv)
            if np.array_equal(hist.PassengerId,ids[n:]):
                by_vector.setdefault(digest(ids[n:],hist.Transported),set()).add(p.parents[1].name)
    pending={}
    for p in (ROOT/'reports').glob('proposal_n27*/manifest.json'):
        manifest=json.loads(p.read_text())
        pending.setdefault(manifest['sha256'],[]).append(p.parent.name)
    inventory={}; selected={}
    def record(key,preds,mode,provenance,keep=False):
        row={'evaluation':mode,'source':provenance,'schemes':{}}
        tq=[]
        for tag,p,b in zip(['random','contiguous'],preds,best):
            assert p.dtype==bool and len(p)==len(ids)
            testp=p[n:];sha=digest(ids[n:],testp)
            tq.append(sha)
            r=dict(sha256=sha,duplicate_official=sorted(set(receipts.get(sha,[]))|by_vector.get(sha,set())),frozen_files=pending.get(sha,[]),parents={})
            for name,base in [('111',first),('16',b),('16_fixed_random_replay',fixed)]:
                d=p!=base;delta=d[:n]*np.where(base[:n]!=y,1,-1)
                r['parents'][name]=dict(test=int(d[n:].sum()),all=count(d[:n],known,base[:n]!=y),fresh=count(d[:n],known&fresh,base[:n]!=y),
                    former_unanimous=count(d[:n],blind,base[:n]!=y),fold_net=[int(delta[known&(f==k)].sum()) for k in range(8)])
            row['schemes'][tag]=r
        row['identical_test_between_protocols']=tq[0]==tq[1]
        inventory[key]=row
        if keep:selected[key]=np.stack(preds)
    # Cache semantics are explicit: archive rows contain edits to a re-calibrated or a fixed parent.
    dynamic=['independent_pool_correction','historical_dissent_screen','manifest_weighted_dissent','rank_passenger_mistakes',
             'unanimous_error_reach','venn_errors','honest_passenger_routing','local_hulls','dissent_likelihood_transfer']
    for name in dynamic:
        path=ART/('nov27_'+name)/'aligned.npz'
        with np.load(path) as z:
            np.testing.assert_array_equal(ids,z['passenger'])
            for k in z.files:
                if not k.startswith('random_'):continue
                stem=k[len('random_'):]
                other='contiguous_'+stem
                if other not in z or z[k].shape!=(len(ids),) or z[k].dtype!=bool:continue
                if name=='dissent_likelihood_transfer' and not stem.endswith(('extra','alternate')):continue
                record(name+'/'+stem,[best[0]^z[k],best[1]^z[other]],'re-calibrated parent and correction',str(path.relative_to(ROOT)))
    fixed_modules=['outsider_directional','crossed_booking_bagging','facility_integer_evidence','manifest_group_dro','manifest_context_coding']
    for name in fixed_modules:
        path=ART/('nov27_'+name)/'aligned.npz'
        with np.load(path) as z:
            np.testing.assert_array_equal(ids,z['ids' if 'ids' in z else 'passenger'])
            for k in z.files:
                if k.startswith('random_'):other='ordered_'+k[7:];stem=k[7:]
                elif k.endswith('_random'):other=k[:-7]+'_ordered';stem=k[:-7]
                else:continue
                if other not in z or z[k].shape!=(len(ids),) or z[k].dtype!=bool:continue
                record(name+'/'+stem,[fixed^z[k],fixed^z[other]],'independent challenger on fixed random parent',str(path.relative_to(ROOT)))
    # Reproduce earlier directional candidates and extend their calibration to contiguous orders.
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:cols=z['scores'][:,-3:]
    with np.load(ART/'nov26_surprise_sampled_foundation/aligned.npz') as z:augmented=z['full']
    with np.load(ART/'nov27_directional_foundation/test.npz') as frozen:
        for nm,prob in dict(Mitra=cols[:,0],TabICL=cols[:,1],Nori=cols[:,2],TabICL_augmented=augmented,median=np.median(cols,axis=1),
                           consensus=np.where(first,cols.max(1),cols.min(1))).items():
            risk=np.where(first,1-prob,prob)
            votes=((cols>=.5)!=first[:,None]).sum(1)
            for shrink in [0,1]:
                pp=[]
                for scheme in ['random','contiguous']:
                    part=splits(g,f,scheme)
                    flag=np.zeros(len(ids),bool);cuts=[]
                    for k in range(8):
                        for j in range(4):
                            t=np.flatnonzero(known&(f==k)&(part!=j));q=np.flatnonzero((f==k)&(part==j))
                            cut=calibrate(risk,first,y,t,shrink)
                            flag[q]=risk[q]>=cut[first[q].astype(int)]
                        cuts.append(calibrate(risk,first,y,np.flatnonzero(known&(f==k)),shrink))
                    flag[n:]=(risk[n:,None]>=np.asarray(cuts)[:,first[n:].astype(int)].T).sum(1)>4
                    pp.append(flag)
                np.testing.assert_array_equal(first[n:]^pp[0][n:],frozen[f'{nm}_held_{shrink}'])
                record(f'directional/{nm}_{shrink}',[first^x for x in pp],'challenger calibrated independently against 111','directional_foundation_transfer_n27.py')
                if shrink==1:
                    record(f'directional/{nm}_confirmed',[first^(x&(votes>=2)) for x in pp],'challenger calibrated independently against 111','directional_foundation_transfer_n27.py')
    # Fixed auxiliary opinions are replayed with each version of the directional parent. This evaluates parent calibration stability only.
    with np.load(ART/'nov27_mitra_fare/aligned.npz') as z:
        np.testing.assert_array_equal(ids,z['ids'])
        for k in [k for k in z.files if k.startswith('111_')]:
            flag=z[k]
            record('Mitra/'+k,[first^flag]*2,'single auxiliary fit; same predictions reused','mitra_fare_evidence_n27.py')
            record('Mitra/retain_'+k,[b^(flag&~m) for b,m in zip(best,locks)],'same auxiliary fit; two calibrations of parent','mitra_fare_evidence_n27.py',True)
    path=ART/'nov27_adjudicate_paid_release_paths'
    with np.load(path/'full_random.npz') as a,np.load(path/'full_ordered.npz') as b,np.load(path/'test_full.npz') as t:
        np.testing.assert_array_equal(t['passenger'],ids[n:])
        for k in a.files:
            flags=[np.r_[v[k],t[k]] for v in (a,b)]
            record('release_paths/111_'+k,[first^v for v in flags],'same-fold distinct order calibration','adjudicate_paid_release_paths_n27.py')
            record('release_paths/retain_'+k,[c^(v&~m) for c,v,m in zip(best,flags,locks)],'both parent and challenger recalibrated','adjudicate_paid_release_paths_n27.py',True)
    # Reconstruct independent archive branches from the original two released binary prediction files.
    from frozen_queue_shared_errors_n26 import paired, RELEASES
    cache=OUT/'archived_releases.npz'
    if not cache.exists():
        OUT.mkdir(exist_ok=True)
        ar,ay,ag,af,ak,matrix=paired()
        for actual,expected in [(ar.PassengerId,ids[:n]),(ay,y),(ag,g),(af,f),(ak,known)]:
            np.testing.assert_array_equal(actual,expected)
        release_test=np.column_stack([test_decisions(label,test.PassengerId)[0] for label in RELEASES.values()])
        np.savez_compressed(cache,ids=ids,fold=f,y=y,names=np.array(list(RELEASES)),pred=np.vstack([matrix,release_test]))
    with np.load(cache) as z:
        np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],f)
        bank={name:z['pred'][:,i] for i,name in enumerate(z['names'])}
    complete=pd.concat([raw,test],ignore_index=True)
    groups=complete.PassengerId.str.split('_').str[0]
    multi=groups.duplicated(keep=False).to_numpy()
    fees=complete[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    paid=fees.gt(0).any(axis=1).to_numpy()
    for scope,mask in [('multi',multi),('unpaid',~paid),('multi_unpaid',multi&~paid)]:
        for method,suggestion in [('fields',(bank['fields']!=bank['nine'])),('band',(bank['band']!=bank['nine'])),
            ('fields_or_band',(bank['fields']!=bank['nine'])|(bank['band']!=bank['nine']))]:
            flag=mask&suggestion&(first==bank['nine'])
            record('archive/'+scope+'_'+method,[b^(flag&~m) for b,m in zip(best,locks)],
                    'same challenger; two calibrations of parent; separate historical models audited in independent_archive_transfer_n27.json',
                    'independent_archive_transfer_n27.py',True)
    for suffix,bank_count in [('one',1),('two',2)]:
        folder=ART/'nov27_complete_receipt_counterfactual'
        with np.load(folder/f'aligned_random_{bank_count}.npz') as za,np.load(folder/f'aligned_ordered_{bank_count}.npz') as zb:
            np.testing.assert_array_equal(za['ids'],ids);np.testing.assert_array_equal(zb['ids'],ids)
            for k in za.files:
                if k not in zb or za[k].dtype!=bool or za[k].shape!=first.shape or k=='anchor':continue
                record('complete_receipt_'+suffix+'/'+k,[fixed^za[k],fixed^zb[k]],'independent random and contiguous source fits on fixed parent','assess_complete_receipt_counterfactual_n27.py')
    # The masked receipt and fifty-person components are already contained in an officially evaluated union.
    # They are inventoried by digest but not eligible as proposals for additional submissions.
    for name in ['proposal_n27_matched_receipt_context','proposal_n27_fifty_fare_adjudication','proposal_n27_three_payment_states','proposal_n27_joint_masked_fare','proposal_n27_unpaid_group_transfer','proposal_n27_independent_tabnet']:
        hist=pd.read_csv(ART/name/'submission.csv')
        np.testing.assert_array_equal(hist.PassengerId,ids[n:]);sha=digest(ids[n:],hist.Transported)
        ref=pending.get(sha,[])
        inventory['file/'+name]=dict(sha256=sha,duplicate_official=sorted(set(receipts.get(sha,[]))|by_vector.get(sha,set())),frozen_files=ref,
              edits_vs_111=int((hist.Transported!=first[n:]).sum()),edits_vs_16=int((hist.Transported!=best[0][n:]).sum()))
    # Distinct signal combination: in a complete paid bill, a model trained with true fare must newly cross the label while its
    # matched raw and shuffled-fare controls both abstain, or a separately calibrated model must support the release path.
    for method in ['real','paths']:
        left=selected['Mitra/retain_111_specific_with_support']
        right=selected['release_paths/retain_'+method]
        candidate=[b^((a!=b)|(c!=b)) for b,a,c in zip(best,left,right)]
        record('combined/Mitra_specific_or_'+method,candidate,'two calibrations; Mitra features shared, path estimates refit','audit_unsubmitted_branches_n27.py',True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'selected.npz',ids=ids,y=y,fold=f,first=first,random_parent=best[0],contiguous_parent=best[1],**selected)
    REPORT.write_text(json.dumps(dict(known=int(known.sum()),fresh=int((known&fresh).sum()),official_receipt_hashes=len(receipts),
        notes=['Both local protocols overlap. They are order-disjoint calibration splits inside the same eight base-model validation folds except where an independently trained challenger is indicated.',
               'Official duplicates are detected by file SHA and by re-serializing aligned PassengerId and boolean vectors when archived CSV is present.',
               'None of the online labels or individual correctness values is reconstructed. The two related masked-receipt and fifty-person frozen files are only inventoried.'],candidates=inventory),indent=2)+'\n')
    def net(m):return m['fixed']-m['harmed']
    for k,v in inventory.items():
        if 'schemes' not in v:continue
        ab=[v['schemes'][s]['parents']['16'] for s in ('random','contiguous')]
        if min(net(a['all']) for a in ab)>0 and all(a['test'] for a in ab):
            print(k,[(net(a['all']),net(a['fresh']),a['test']) for a in ab],flush=True)
    print(len(inventory),flush=True)


if __name__=='__main__':run()
