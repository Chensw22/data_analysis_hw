"""Post-prediction arithmetic audit and descriptive error attribution; no fitting."""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'
import json
import numpy as np
from sklearn.metrics import log_loss
from limix2_setup import ROOT,sha,write
import error_overlap_diagnostic as overlap
from exception_error_detector_nov18 import reporting_masks
from limix2_planet_trial import choose_support


def main():
    folder=ROOT/'reports/limix2_planet_trial';out=ROOT/'artifacts/limix2_planet_trial'
    result=json.loads((folder/'result.json').read_text());plan=json.loads((folder/'protocol.json').read_text())
    if result['status']!='complete_fixed3fold_cpu_planet_trial':raise RuntimeError('Need complete predictions.')
    for name,digest in {**plan['source_hashes'],**result['evidence_sha256']}.items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Changed source/input: '+name)
    inputs,splits,folds,bases,evidence=overlap.load_authenticated();dev,y=inputs['dev'],inputs['y']
    frame=inputs['features'].iloc[dev].reset_index(drop=True);planets=frame.HomePlanet.astype(str).to_numpy()
    masks=reporting_masks(dev,folds,y,inputs,bases,evidence)
    if sha(out/'oof.npz')!=result['oof_sha256']:raise RuntimeError('Prediction hash differs.')
    with np.load(out/'oof.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['dev_ids'],dev);np.testing.assert_array_equal(z['fold_id'],folds)
        np.testing.assert_array_equal(z['raw'],bases['rawMitra'])
        if z['passenger_ids'].dtype.kind!='U':raise RuntimeError('Nonportable IDs.')
        ps={name:z[key].copy() for name,key in [('LimiX2','probability'),('rawMitra','raw'),('v96','v96'),('half_rawMitra','half_raw'),('half_v96','half_v96')]}
    support_checks=[]
    for fold,(a,b) in enumerate(splits):
        packet=json.loads((out/f'fold{fold}_packet.json').read_text());seen=[]
        for route,part in packet['routes'].items():
            expected=choose_support(a,planets,route,20261118+fold);sid=np.asarray(part['support_ids'])
            np.testing.assert_array_equal(sid,dev[expected]);np.testing.assert_array_equal(part['support_y'],y[expected])
            positions=np.flatnonzero(planets[b]==route);qids=dev[b[positions]]
            np.testing.assert_array_equal(part['query_ids'],qids);np.testing.assert_array_equal(part['positions'],positions)
            if set(sid)&set(dev[b]) or len(set(sid))!=2048:raise RuntimeError('Invalid support boundary.')
            with np.load(out/f'fold{fold}_{route}_prediction.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['query_ids'],qids);np.testing.assert_array_equal(z['support_ids'],sid)
                np.testing.assert_array_equal(z['prob'],ps['LimiX2'][b[positions]])
            seen.extend(qids.tolist())
            support_checks.append(dict(fold=fold,route=route,support_rows=len(sid),query_rows=len(qids),train_only=True))
        if sorted(seen)!=sorted(dev[b].tolist()):raise RuntimeError('Query partition differs.')
    rawok=(ps['rawMitra']>=.5)==y
    def describe(p,mask):
        ok=(p>=.5)==y;repair=mask&ok&~rawok;damage=mask&~ok&rawok
        return dict(rows=int(mask.sum()),correct=int((ok&mask).sum()),repairs=int(repair.sum()),breaks=int(damage.sum()),
            negative_repairs=int((repair&(y==0)).sum()),positive_repairs=int((repair&(y==1)).sum()),
            new_false_positives=int((damage&(y==0)).sum()),new_false_negatives=int((damage&(y==1)).sum()),
            mean_raw_confidence_on_changes=float(np.maximum(ps['rawMitra'],1-ps['rawMitra'])[repair|damage].mean()) if (repair|damage).any() else None,
            mean_candidate_confidence_on_changes=float(np.maximum(p,1-p)[repair|damage].mean()) if (repair|damage).any() else None)
    bills=frame[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    missing=bills.isna().any(axis=1).to_numpy();zero=(bills.sum(axis=1).to_numpy()==0)&~missing
    groupings={'planet_cryo':np.char.add(np.char.add(planets.astype(str),'|'),frame.CryoSleep.astype(str).to_numpy().astype(str)),
        'spending':np.where(missing,'incomplete',np.where(zero,'complete_zero','complete_positive')),
        'age':np.where(frame.Age.isna(),'missing',np.where(frame.Age<=12,'<=12',np.where(frame.Age<18,'13-17','>=18'))),
        'deck':frame.Deck.astype(str).to_numpy()}
    stats={}
    for name,p in ps.items():
        row=describe(p,np.ones(len(y),dtype=bool));row.update(accuracy=float(np.mean((p>=.5)==y)),log_loss=float(log_loss(y,p)))
        if abs(row['accuracy']-result['metrics'][name]['accuracy'])>1e-14 or abs(row['log_loss']-result['metrics'][name]['log_loss'])>1e-14:raise RuntimeError('Reported metric mismatch.')
        row['historical_subsets']={k:describe(p,m) for k,m in masks.items()}
        row['slices']={k:{str(value):describe(p,values==value) for value in sorted(set(values))} for k,values in groupings.items()}
        row['folds']=[describe(p,folds==f) for f in range(3)]
        stats[name]=row
    oldpath=ROOT/'artifacts/limix2_passenger_trial/oof.npz'
    oldresult=json.loads((ROOT/'reports/limix2_passenger_trial/result.json').read_text())
    if sha(oldpath)!=oldresult['oof_sha256']:raise RuntimeError('First-batch hash differs.')
    with np.load(oldpath,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['dev_ids'],dev);old=z['probability'].copy()
    hard=masks['all_recipes_common_wrong546'];now=((ps['LimiX2']>=.5)==y)&hard;before=((old>=.5)==y)&hard
    audit=dict(status='verified_saved_probabilities_and_all_route_boundaries',metrics=stats,support_checks=support_checks,
        hard546_batch_comparison=dict(first512_correct=int(before.sum()),planet2048_correct=int(now.sum()),both_correct=int((before&now).sum()),newly_correct=int((~before&now).sum()),lost=int((before&~now).sum())),
        evidence=dict(result_sha256=sha(folder/'result.json'),oof_sha256=sha(out/'oof.npz'),analysis_source_sha256=sha(__file__),first_batch_oof_sha256=sha(oldpath)),
        limitations=['Slices are posthoc descriptions, not causal mechanisms, learned routing, or label quality diagnoses.',
                    'Context size, pipeline count and planet routing changed together; no single-factor attribution.',
                    'Historical hard masks were absent from support selection and model inference.'])
    write(folder/'error_analysis.json',audit)
    print(json.dumps({'status':audit['status'],'hard546_batch_comparison':audit['hard546_batch_comparison'],
        'metrics':{k:{n:v[n] for n in ['accuracy','log_loss','repairs','breaks']} for k,v in stats.items()}},indent=2))


if __name__=='__main__':main()
