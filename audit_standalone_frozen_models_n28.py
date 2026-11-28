"""Compare archived full-passenger model predictions to the four tied releases on identical labeled rows."""
import hashlib
import json

import numpy as np
import pandas as pd

from archive_source_harm_veto_n28 import inputs
from corroborated_booking_allocation_n28 import final_decision
from repeated_archive_misses_n26 import ART, ROOT, cohort, count
from unanimous_error_reach_n27 import inputs as architecture_inputs

OUT = ART / 'nov28_frozen_full_model_inventory'


def run():
    frame,y,groups,fold,ids,names,p,seven,versions=inputs()
    n=len(y)
    parent=versions['random'][:,0]
    _,_,known,fresh=cohort(frame.iloc[:n],y,fold)
    wrong=parent[:n]!=y
    assert known.sum()==8532
    def read(path):
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z['ids' if 'ids' in z else 'passenger'],ids)
            for key,desired in [('y',y),('fold',fold),('f',fold)]:
                if key in z:np.testing.assert_array_equal(z[key],desired)
            return {k:z[k] for k in z}
    by_version={}
    by_sha={}
    for receipt in (ROOT/'reports').glob('*/online_attempts/*.json'):
        record=json.loads(receipt.read_text())
        name=receipt.parents[1].name
        if 'file_sha256' in record:by_sha.setdefault(record['file_sha256'],set()).add(name)
        path=ART/name/'submission.csv'
        if not path.is_file() or record.get('evaluation',{}).get('Status')!='COMPLETE':continue
        if record.get('file_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():continue
        official=pd.read_csv(path)
        np.testing.assert_array_equal(official.PassengerId,ids[n:])
        by_version[name]=(official.Transported.to_numpy(),record['evaluation']['Public Score'])
    audit={}
    def record(key,random,ordered=None):
        schemes={'random':random}
        if ordered is not None:schemes['ordered']=ordered
        entry={}
        for mode,hard in schemes.items():
            assert hard.shape==ids.shape and hard.dtype==bool
            agree=(versions[mode]==parent[:,None]).all(1)
            changes=hard!=parent
            distances=sorted((int((other!=hard[n:]).sum()),name,score) for name,(other,score) in by_version.items())
            payload=pd.DataFrame({'PassengerId':ids[n:],'Transported':hard[n:]}).to_csv(index=False).encode()
            sha=hashlib.sha256(payload).hexdigest()
            entry[mode]=dict(all=count(changes[:n],known,wrong),fresh=count(changes[:n],known&fresh,wrong),
                    four=count(changes[:n],known&agree[:n],wrong),edits=int(changes[n:].sum()),
                    four_edits=int((changes[n:]&agree[n:]).sum()),nearest=distances[0],
                    official_by_sha=sorted(by_sha.get(sha,[])),sha256=sha)
        audit[key]=entry
    for i,name in enumerate(names):record('historic/'+name,p[:,i]>=.5)
    z=read('nov26_latent_passenger_difficulty/panel.npz')
    for j,name in enumerate(z['names']):record('panel/'+name,z['scores'][:,j]>=.5)
    ar=architecture_inputs()
    np.testing.assert_array_equal(ar[1].PassengerId,ids)
    for name,prob in ar[-1].items():
        if name!='random_environment_boost':record('architecture/'+name,prob>=.5)
    for j,name in enumerate(['historic median','Mitra','TabICL fare','Nori fare','foundation median','TabDPT','TabNet median']):
        record('ordinary/'+name,seven[:,j]>=.5)
    z=read('nov27_outsider_directional/aligned.npz')
    for j in range(z['tabnet'].shape[1]):record(f'TabNet/view_{j}',z['tabnet'][:,j]>=.5)
    z=read('nov26_surprise_sampled_foundation/aligned.npz')
    for key in ['pilot','unadjusted','adjusted','uniform','full']:record('surprise/'+key,z[key]>=.5)
    for family,file in [('field_pair_evidence','aligned.npz'),('manifest_tree_mixture','scores.npz'),
                        ('shared_interactions','predictions.npz'),('solo_corridor_receipts','aligned.npz')]:
        z=read('nov27_'+family+'/'+file)
        for j in range(3):
            if family=='solo_corridor_receipts' and j==2:continue
            record(f'{family}/view_{j}',z['random'][:,j]>=.5,z['ordered'][:,j]>=.5)
    z=read('nov28_complete_architecture_committee/scores.npz')
    for j,key in enumerate(['full','without_classic','without_foundation','without_structured']):
        record('complete_committee/'+key,z['random_scores'][:,j]>=0,z['contiguous_scores'][:,j]>=0)
    booking=read('nov27_conditional_party_allocation/aligned.npz')
    for j,key in enumerate(['ordinary','conditioned']):
        record('booking/'+key,booking['random_scores'][:,j]>=0,booking['ordered_scores'][:,j]>=0)
    archive=read('nov27_unsubmitted_audit/archived_releases.npz')
    for i,name in enumerate(archive['names']):record('old_releases/'+name,archive['pred'][:,i])
    def released(name,first,second):
        path=ART/name/'submission.csv'
        actual=pd.read_csv(path)
        np.testing.assert_array_equal(actual.PassengerId,ids[n:])
        np.testing.assert_array_equal(actual.Transported,first[n:])
        np.testing.assert_array_equal(actual.Transported,second[n:])
        record('frozen_csv/'+name,first,second)
    flags=[final_decision(parent,z[key+'_scores'],seven,booking[alloc+'_supported_increment'])
           for key,alloc in [('random','random'),('contiguous','ordered')]]
    released('proposal_n28_corroborated_booking_allocation',*(parent^f for f in flags))
    party=read('nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz')
    released('proposal_n27_unanimous_party_experts',*(parent^party[k] for k in ('random','ordered')))
    outsider=read('nov27_outsider_directional/aligned.npz')
    for policy in ['spread_split_anchored','spread_retained_anchored']:
        first,second=(parent^outsider[policy+'_'+k] for k in ['random','ordered'])
        if policy=='spread_split_anchored':
            released('proposal_n27_architecture_increment',first,second)
        else:record('archived_spread/'+policy,first,second)
    OUT.mkdir(exist_ok=True)
    report=dict(rows=int(known.sum()),direction_correct=int((known&~wrong).sum()),
        archived_csv=sum(1 for _ in ART.glob('*/submission.csv')),online_vectors=len(by_version),
        receipt_hashes=len(by_sha),models=audit)
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/nov23_matched_learning_curve/frozen_full_model_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    def net(row,key='all'):
        v=row[key];return v['fixed']-v['harmed']
    full=[(net(row['random']),key,row['random']['all'],row['random']['four'],row['random']['edits'])
          for key,row in audit.items() if key.split('/')[0] not in ('frozen_csv','archived_spread')]
    print('full model best',sorted(full,reverse=True)[:6])
    for k,r in audit.items():
        if k.split('/')[0] in ('frozen_csv','archived_spread'):
            print(k,[(s,v['all'],v['four'],v['fresh'],v['edits'],v['four_edits'],v['nearest']) for s,v in r.items()])


if __name__=='__main__':
    run()
