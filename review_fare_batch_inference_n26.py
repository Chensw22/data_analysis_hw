"""Independently reconstruct the two frozen files from source model outputs and compare score drift."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from audit_fare_batch_inference_n26 import ART, F, FARE, FIRST, N, OUT, REPORT, SECOND, compare, roster
from held_expert_portfolios_n26 import SIX, panel_predictions, sources
from nori_booking_regression_n26 import packet_dir


def stages(anchor,raw,fine,six,votes,eligible):
    a=anchor.astype(bool)
    f=np.where(eligible&((raw>=.5)==a)&((fine[:,0]>=.5)!=a),fine[:,0],a)
    h=f>=.5
    six_ok=((six>=.5)!=h[:,None]).all(1)&((votes!=h[:,None]).sum(1)>=10)
    base=np.where(eligible&six_ok,six.mean(1),f)
    prior=base>=.5
    edit=eligible & (prior==a) & ((fine>=.5)!=prior[:,None]).all(1)
    second=np.where(edit,np.clip(fine,0,1).mean(1),base)
    return base,second


def profile(p):
    return dict(rows=len(p),outside_0_1=int(((p<0)|(p>1)).sum()),quantile_0_10_50_90_100=np.quantile(p,[0,.1,.5,.9,1]).tolist(),
                predicted_positive=float((p>=.5).mean()),within_0_025_of_threshold=int((abs(p-.5)<=.025).sum()))


def main():
    a,fr,y,g,f=roster();n=len(a);ids=fr.PassengerId.to_numpy(str)[n:]
    def verify(name,vec):
        stored=pd.read_csv(ART/name/'submission.csv')
        np.testing.assert_array_equal(stored.PassengerId,ids)
        np.testing.assert_array_equal(stored.Transported,vec)
        csv=pd.DataFrame(dict(PassengerId=ids,Transported=vec)).to_csv(index=False).encode()
        digest=hashlib.sha256(csv).hexdigest()
        assert digest==hashlib.sha256((ART/name/'submission.csv').read_bytes()).hexdigest()
        assert digest==json.loads((ART/name/'manifest.json').read_text())['sha256']
        return digest
    anchor=pd.read_csv(ART/'proposal_n24_selected_dissent/submission.csv').Transported.to_numpy()
    verify('proposal_n24_selected_dissent',anchor)
    with np.load(ART/'nov26_tabicl_consensus_review/full.npz') as z: names=z['names'][:18]
    with np.load(ART/'proposal_n26_dispute_band/frozen_votes.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        pool=np.array([v.split('/')[0] for v in z['names']]);votes=np.column_stack([z['p'][:,pool==v].mean(1)>=.5 for v in names])
    six=[]
    for arch,fields in [('tree',['register_all','register_paid','sequence_paid','tree_route','swap']),('neural',['neural_route'])]:
        with np.load(ART/'proposal_n26_six_way_dispute'/f'{arch}.npz') as z:
            np.testing.assert_array_equal(z['passenger'],ids)
            six.extend(z[k] for k in fields)
    six=np.column_stack(six)
    with np.load(FARE) as z:
        np.testing.assert_array_equal(z['passenger'],ids)
        raw,fine=z['raw'],z['fine']
    fees=fr[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
    enabled=(fees.notna().all(axis=1)&fees.gt(0).any(axis=1)).to_numpy()
    with np.load(F/'deck_estimates.npz') as z:np.testing.assert_array_equal(z['eligible'],enabled)
    rt={}
    for key in ['own','raw']:
        with np.load(N/f'{key}_test.npz') as z:
            np.testing.assert_array_equal(z['passenger'],ids);np.testing.assert_array_equal(z['labels'],y)
            rt[key]=z['p']
    test_scores={'tabicl_raw':raw,'tabicl_fine':fine,'nori_raw':rt['raw'],'nori_fine':rt['own']}
    train_scores={name:np.zeros(n) for name in test_scores}
    for k in range(8):
        q=np.flatnonzero(f==k);t=np.flatnonzero(f!=k)
        for file,fields in [(F/f'neural_full_{k}.npz',{'tabicl_raw':'raw','tabicl_fine':'fine'}),
                            (N/f'own{k}.npz',{'nori_fine':'p'}),(packet_dir(0)/f'fold{k}.npz',{'nori_raw':'p'})]:
            with np.load(file) as z:
                np.testing.assert_array_equal(z['q'],q);np.testing.assert_array_equal(z['t'],t)
                for key,col in fields.items():train_scores[key][q]=z[col]
    params=(anchor,raw,np.column_stack([fine,rt['own']]),six,votes,enabled[n:])
    baseline,end=stages(*params)
    outputs={}
    for directory,p in [(FIRST,baseline),(SECOND,end)]:
        np.testing.assert_array_equal(p,pd.read_csv(directory/'test_probabilities.csv',float_precision='round_trip').probability)
        outputs[directory.name]=dict(sha256=verify(directory.name,p>=.5),edits=int(((p>=.5)!=anchor).sum()))
    results=dict(outputs=outputs,score_profiles={},inference={},paired_batch={},
        model_inputs={'TabICL':dict(raw_columns=15,fare_columns=25,model_device='cpu',score='class True probability at index 1',cutoff=.5),
                      'Nori':dict(raw_columns=15,fare_columns=24,model_device='cpu',score='regression on 0 or 1',cutoff=.5)},
        comparisons={'test':'Same full labeled support; compare a fixed booking-selected 1,057-passenger test subset alone versus in the whole 4,277-passenger query.',
                     'fold0':'Same held-out query passengers and labeled support; append unlabeled test features to reach 4,277 queries.',
                     'support':'Fixed 1,057 test passengers and identical seed; remove one whole training fold (about 1/8) and refit affected fare scores. Other experts and upstream public baseline remain fixed. This measures decision reproducibility; no test outcomes enter the comparison.'})
    for name in train_scores:
        results['score_profiles'][name]={}
        for tag,p,scope in [('OOF',train_scores[name],enabled[:n]),('test',test_scores[name],enabled[n:])]:
            results['score_profiles'][name][tag]=dict(all=profile(p),eligible=profile(p[scope]))
    for key in ['test','fold0','support0','support1']:
        available={}
        for name in train_scores:
            path=OUT/f'{name}_{key}.npz'
            if not path.exists():continue
            with np.load(path) as z:
                if key!='fold0':np.testing.assert_array_equal(z['full_ids'],ids)
                else:np.testing.assert_array_equal(z['full_ids'],a.PassengerId[f==0])
                stats=dict(comparison=compare(z['paired'],z['changed']),
                    rank_spearman=float(spearmanr(z['paired'],z['changed']).statistic),seconds=float(z['elapsed']))
                if not key.startswith('support'):
                    stats['full_replay']=compare(z['full'],z['reference'])
                results['inference'][name+'_'+key]=stats
                available[name]=(z['rows'],z['paired'],z['changed'])
        if not available:continue
        rows=next(iter(available.values()))[0]
        for r,_,_ in available.values():np.testing.assert_array_equal(rows,r)
        if key!='fold0':
            ii=rows-n;src=params
        else:
            from grouped_peer_teacher_n25 import OUT as PEER
            from full_roster_booking_validation_n23 import ROOT
            with np.load(PEER/'release0.npz') as z:
                np.testing.assert_array_equal(z['q'],rows);low,high=z['nine_on_low_branch'],z['nine_on_high_branch']
            with np.load(ART/'nov26_patched_cabins/known_0_1_2_3_4_5_6_7.npz') as z:
                prior=z['prior'][rows]
            m=prior>=0
            panel=sources()[-1]['full'];p=panel_predictions(panel,'full')
            src=(np.where(prior==1,high,low),train_scores['tabicl_raw'][rows],
                 np.column_stack([train_scores['tabicl_fine'][rows],train_scores['nori_fine'][rows]]),
                 np.column_stack([p[n][rows] for n in SIX]),panel['votes'][rows,:18],enabled[rows])
            ii=np.flatnonzero(m)
        if key!='fold0':args=[v[ii].copy() for v in src];positions=np.arange(len(ii))
        else:args=[v[ii].copy() for v in src];positions=ii
        old=stages(*args)
        for name,(_,_,changed) in available.items():
            if name=='tabicl_raw':args[1]=changed[positions]
            elif name!='nori_raw':args[2][:,int(name=='nori_fine')]=changed[positions]
        after=stages(*args)
        info=dict(rows=len(ii),components=list(available),updated_predictions={})
        for j,release in enumerate(['111','161']):
            x,z=old[j]>=.5,after[j]>=.5
            champ=args[0].astype(bool)
            d={'changed':int((x!=z).sum()),'old_edits':int((x!=champ).sum()),'new_edits':int((z!=champ).sum()),
               'edits_retained':int(((x!=champ)&(z!=champ)).sum())}
            if key=='fold0':
                truth=y[rows[ii]]
                d.update(fixed=int(((x!=truth)&(z==truth)).sum()),harmed=int(((x==truth)&(z!=truth)).sum()))
            info['updated_predictions'][release]=d
        results['paired_batch'][key]=info
    REPORT.write_text(json.dumps(results,indent=2)+'\n')
    print(results['outputs'],results['paired_batch'])


if __name__=='__main__':
    main()
