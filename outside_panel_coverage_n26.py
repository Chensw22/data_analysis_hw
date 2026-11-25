"""Measure additional outsider votes only on focal predictions that exclude the entire focal order."""
import json
import numpy as np

from full_roster_booking_validation_n23 import OUT as VALIDATION, ROOT, data
from grouped_limix_receipt_n26 import OUT as LIMIX
from held_expert_portfolios_n26 import DECK, outcome_scores, panel_predictions, replay, sources, splits
from limix2_setup import sha

ART = ROOT/'artifacts'
REPORT = ROOT/'reports/nov23_matched_learning_curve/outside_panel_coverage.json'
MITRA = ['e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19',
         '5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09']
TABDPT = '97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'


def projection():
    raw,y,g,f = data()
    _,_,_,panels = sources()
    v = panels['full']
    np.testing.assert_array_equal(v['ids'],np.arange(len(y)))
    np.testing.assert_array_equal(v['fold'],f)
    p = panel_predictions(v,'full')
    original = outcome_scores(v,p)>=.5
    assert original.shape==(len(y),99)
    valid = np.zeros(len(y),bool)
    parent = np.zeros(len(y),bool)
    with np.load(DECK/'deck_estimates.npz') as z:
        enabled = z['eligible']
    views = {}
    for key,src,dst,t,q in splits(panels,g):
        if key.startswith('bookingmix'):
            assert not valid[q].any()
            parent[q] = replay(key,q,dst,p,enabled)
            valid[q] = True
        elif key=='historical_to_new':
            current = parent.copy()
            current[q] = replay(key,q,dst,p,enabled)
            views[key] = (np.isin(np.arange(len(y)),q),current)
    fresh = ~np.isin(g,g[panels['historical']['ids']])
    views['bookingmix'] = valid,parent
    views['fresh_bookingmix'] = valid&fresh,parent
    assert valid.sum()==6055 and (valid&fresh).sum()==1033
    return raw,y,g,f,views,original


def validate(z,t,q,y,g,fields):
    for key,expected in zip(fields,(t,q,y[t]),strict=True):
        np.testing.assert_array_equal(z[key],expected)
    assert not np.isin(g[t],g[q]).any()


def archives(raw,y,g,fold,with_limix):
    columns = {}
    def put(name,q,p):
        dest = columns.setdefault(name,np.full(len(y),np.nan))
        assert not np.isfinite(dest[q]).any() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        dest[q] = p
    for k in map(int,np.unique(fold)):
        t,q = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
        with np.load(VALIDATION/f'mitra96_fold_{k}.npz') as z:
            validate(z,t,q,y,g,('train','query','training_labels'))
            assert z['checkpoint']==MITRA[1] and z['steps']==15
            put('Mitra2_adapted',q,z['p'])
        with np.load(ART/f'nov25_full_context_generations/eight_{k}.npz') as z:
            validate(z,t,q,y,g,('t','q','labels'))
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
            np.testing.assert_array_equal(z['checkpoints'],MITRA)
            put('Mitra1_frozen',q,z['v1'])
            put('Mitra2_frozen',q,z['v2'])
        with np.load(ART/f'nov25_cross_architecture_state/{k}_global.npz') as z:
            validate(z,t,q,y,g,('t','q','training_labels'))
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
            assert z['checkpoint']==TABDPT
            put('TabDPT',q,z['prob'])
        with np.load(ART/f'nov25_passenger_joint_likelihood/fold{k}.npz') as z:
            validate(z,t,q,y,g,('t','q','train_y'))
            np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
            put('mask_net_discriminative',q,z['outcome_only'])
            put('mask_net_joint',q,z['direct'])
            put('mask_net_class_likelihood',q,(z['chain_0']+z['chain_1'])/2)
        for folder,key,name in [('conditional_bill_flow',2,'conditional_bill_flow'),
                                ('latent_profiles',1,'latent_class_mixture'),
                                ('conditional_profiles',1,'conditional_profile_mixture')]:
            with np.load(ART/f'nov24_{folder}/full_{k}.npz') as z:
                validate(z,t,q,y,g,('t','q','truth'))
                put(name,q,z['p'][:,key])
        if with_limix:
            path = LIMIX/f'fold{k}_prediction.npz'
            packet_path = LIMIX/f'fold{k}_packet.json'
            packet = json.loads(packet_path.read_text())
            worker = json.loads((LIMIX/f'fold{k}_worker.json').read_text())
            assert sha(path)==worker['prediction_sha256'] and sha(packet_path)==worker['packet_sha256']
            with np.load(path) as z:
                support = z['support_ids']
                np.testing.assert_array_equal(z['query_ids'],q)
                np.testing.assert_array_equal(packet['support_ids'],support)
                np.testing.assert_array_equal(packet['support_y'],y[support])
                assert len(np.unique(support))==512 and np.isin(support,t).all()
                put('LimiX2',q,z['prob'])
    assert all(np.isfinite(p).all() for p in columns.values())
    # EXAONE used a distinct grouped split on 6,954 development passengers. It can only
    # be scored where the actual archived held-out passenger exists, and is reported separately.
    folder = ART/'exaone_group_20261117'
    report = ROOT/'reports/exaone_group_20261117'
    checksums=json.loads((report/'summary.json').read_text())['result_files_sha256']
    partial = np.full(len(y),np.nan)
    prediction = folder/'EXAONE_v91_CAST_default8_seed20261112.npz'
    assert sha(prediction)==checksums[prediction.name]
    with np.load(prediction) as z:
        rows,hf,p = z['dev_ids'],z['fold_id'],z['prob']
    for k in np.unique(hf):
        path=folder/f'fold{k}_rows.npz'
        assert sha(path)==checksums[path.name]
        with np.load(path) as z:
            t,q = z['train_ids'],z['validation_ids']
            np.testing.assert_array_equal(q,rows[hf==k])
            np.testing.assert_array_equal(t,rows[hf!=k])
            assert not np.isin(g[t],g[q]).any()
            partial[q]=p[hf==k]
    if with_limix:
        path=ART/'nov26_exaone_remaining_orders/scores.npz'
        q=np.flatnonzero(~np.isfinite(partial))
        t=rows[~np.isin(g[rows],g[q])]
        with np.load(path) as z:
            validate(z,t,q,y,g,('t','q','train_y'))
            assert z['checkpoint']=='2aa723e024e5c56c778d6c79cc3cee7675d1b6a03c06fa08c80e620e228b224d'
            full=partial.copy()
            full[q]=z['p']
            assert np.isfinite(full).all() and ((full>=0)&(full<=1)).all()
            columns['EXAONE_grouped']=full
    return columns,partial


def count(mask,wrong,flip):
    return dict(rows=int(mask.sum()),errors=int((mask&wrong).sum()),fixed=int((mask&wrong&flip).sum()),
                harmed=int((mask&~wrong&flip).sum()))


def run(with_limix=True):
    raw,y,g,fold,views,original=projection()
    scores,exaone=archives(raw,y,g,fold,with_limix)
    names=list(scores)
    p=np.column_stack(list(scores.values()))
    # Reject silent duplicates of the 99 archived votes; related model families still get one vote in committees.
    identical={name:np.flatnonzero(((p[:,j,None]>=.5)==original).all(0)).tolist() for j,name in enumerate(names)}
    assert not any(identical.values())
    by_family={'Mitra':['Mitra1_frozen','Mitra2_frozen','Mitra2_adapted'],'TabDPT':['TabDPT'],
               'mask_net':['mask_net_discriminative','mask_net_joint','mask_net_class_likelihood'],
               'bill_flow':['conditional_bill_flow'],'profiles':['latent_class_mixture','conditional_profile_mixture']}
    if with_limix:
        by_family['LimiX2']=['LimiX2']
        by_family['EXAONE']=['EXAONE_grouped']
    report={'complete_model_names':names,'family_votes':by_family,
            'same_as_existing_vote_entire_roster':identical,
            'partial_exaone_protocol':'Whole-order historical 3-fold predictions, 6954 development passengers; only measured overlap reported.',
            'completed_exaone_protocol':'Supplement the old grouped predictions with one 5835-passenger fit excluding all orders represented among the remaining 1739 passengers. No query contributes support labels.',
            'LimiX2_recipe':'Built with StableAI LimiX. One official pipeline on 512 stratified supports per outer fold.'}
    for tag,(valid,parent) in views.items():
        wrong=parent!=y
        all99=(original==parent[:,None]).all(1)
        hard=valid&all99
        if tag=='bookingmix':
            assert (hard&wrong).sum()==136
        elif tag=='historical_to_new':
            assert (hard&wrong).sum()==26
        discord=(p>=.5)!=parent[:,None]
        families=list(by_family)
        fam_p=np.column_stack([p[:,[names.index(n) for n in by_family[f]]].mean(1) for f in families])
        fam_d=(fam_p>=.5)!=parent[:,None]
        pretrained_names=[f for f in ('Mitra','TabDPT','LimiX2','EXAONE') if f in families]
        pretrained=[families.index(f) for f in pretrained_names]
        actions={n:discord[:,j] for j,n in enumerate(names)}
        actions.update({'any_frozen_or_adapted_transformer':discord[:,[names.index(n) for f in pretrained_names for n in by_family[f]]].any(1),
                        'any_raw_archived_or_new_vote':discord.any(1),
                        'two_pretrained_families':fam_d[:,pretrained].sum(1)>=2,
                        'all_pretrained_families':fam_d[:,pretrained].all(1),
                        'two_independent_family_votes':fam_d.sum(1)>=2,
                        'three_independent_family_votes':fam_d.sum(1)>=3,
                        'majority_of_families':fam_d.sum(1)>len(families)/2,
                        'two_opposing_at_least_70pct':(np.where(parent[:,None],1-fam_p,fam_p)>=.7).sum(1)>=2})
        scopes={'all':valid,'all99_support_parent':hard,'all99_negative':hard&~parent,'all99_positive':hard&parent}
        # Histograms by passenger indicate whether potentially correct models identify the same people.
        result={n:{k:count(mask,wrong,d) for k,mask in scopes.items()} for n,d in actions.items()}
        result['EXAONE_partial']={k:count(mask&np.isfinite(exaone),wrong,(exaone>=.5)!=parent) for k,mask in scopes.items()}
        result['hard_errors_reached_only_by_this_column']={n:int((hard&wrong&d&(discord.sum(1)==1)).sum()) for n,d in actions.items() if n in names}
        result['hard_error_overlap']={}
        for j,name in enumerate(names):
            result['hard_error_overlap'][name]={
                other:int((hard&wrong&discord[:,j]&discord[:,k]).sum()) for k,other in enumerate(names)}
        result['uncovered_error_profile']={}
        solo=raw.PassengerId.groupby(g).transform('size').to_numpy()==1
        paid=raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).any(axis=1).to_numpy()
        for category,mask in [('solo',solo),('with_peer',~solo),('some_payment',paid),('no_visible_payment',~paid),
                              ('predicted_stayed',~parent),('predicted_transported',parent)]:
            result['uncovered_error_profile'][category]=count(hard&mask,wrong,discord.any(1))
        without_exaone=discord[:,[n!='EXAONE_grouped' for n in names]].any(1)
        result['without_EXAONE']=count(hard,wrong,without_exaone)
        result['increment_only_from_EXAONE']=count(hard,wrong,discord.any(1)&~without_exaone)
        if tag=='bookingmix':
            size=raw.PassengerId.groupby(g).transform('size').to_numpy()
            fees=raw[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
            result['hard_cases_posthoc']=[dict(passenger=str(raw.PassengerId.iloc[i]),booking_size=int(size[i]),
                observed_spend=float(fees.iloc[i].sum()),active_services=int((fees.iloc[i]>0).sum()),
                verified_correct_columns=[n for j,n in enumerate(names) if discord[i,j]])
                for i in np.flatnonzero(hard&wrong)]
        report[tag]=result
        print(tag,len(names),result['any_raw_archived_or_new_vote']['all99_support_parent'],flush=True)
    if with_limix:
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    run()
