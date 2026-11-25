"""Select fixed dissenters on the historical cohort and evaluate later, nonoverlapping solo bookings."""
import json

import numpy as np
from threadpoolctl import threadpool_limits

from attentive_tabnet_n26 import MODELS
from audit_legacy_queue_reach_n26 import contemporary
from expanded_joint_error_witnesses_n26 import FAMILIES,load_scores
from held_expert_portfolios_n26 import NEW,outcome_scores,panel_predictions,sources
from outside_panel_coverage_n26 import archives
from price_implied_deck_n26 import FEES
from solo_fixed_expert_pairs_n26 import choose,metrics
from solo_remaining_countervotes_n26 import context
from tabnet_attention_transfer_n26 import historical_outputs
from target_mix_expert_routing_n26 import ROOT

ART=ROOT/'artifacts'
REPORT=ROOT/'reports/nov23_matched_learning_curve/historical_solo_expert_transfer.json'
CHOSEN=['original_99/permutation_selection/selected','original_99/service_order_mixture/empty']


def verify_named_pair(tag,ids,fold,raw,y,g,bits,names):
    for k in np.unique(fold):
        m=fold==k
        q=ids[m]
        t=ids[(~m)&~np.isin(g[ids],g[q])]
        for folder,key,col,labels,alias in [
                ('permutation_selection','prob',1,'labels',CHOSEN[0]),('service_order_mixture','p',1,'truth',CHOSEN[1])]:
            with np.load(ART/f'nov24_{folder}/{tag}_{k}.npz') as z:
                for name,expected in [('train',t),('query',q),(labels,y[t])]:
                    np.testing.assert_array_equal(z[name],expected)
                np.testing.assert_array_equal(z[key][:,col]>=.5,bits[m,names.index(alias)])
        assert not np.isin(g[t],g[q]).any()


def archived_outsiders(raw,y,g,ids,fold):
    with np.load(ART/'v96_oof_completion/oof.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'],ids)
        np.testing.assert_array_equal(z['passenger_ids'],raw.PassengerId.iloc[ids])
        out={'Mitra2_adapted':z['prob']}
    for k in np.unique(fold):
        m=fold==k
        q=ids[m]
        records=[('nov25_full_context_generations/historical_','labels',['Mitra1_frozen','Mitra2_frozen'],['v1','v2']),
                 ('nov25_passenger_joint_likelihood/historical','train_y',['mask_net_discriminative','mask_net_joint','mask_net_class_likelihood'],
                  ['outcome_only','direct',('chain_0','chain_1')]),
                 ('nov24_conditional_bill_flow/historical_','truth',['conditional_bill_flow'],[2]),
                 ('nov24_latent_profiles/historical_','truth',['latent_class_mixture'],[1]),
                 ('nov24_conditional_profiles/historical_','truth',['conditional_profile_mixture'],[1])]
        for folder,label,keys,cols in records:
            with np.load(ART/f'{folder}{k}.npz') as z:
                t=z['t']
                np.testing.assert_array_equal(z['q'],q)
                assert np.isin(t,ids[~m]).all()
                np.testing.assert_array_equal(z[label],y[t])
                if 'passenger' in z:
                    np.testing.assert_array_equal(z['passenger'],raw.PassengerId.iloc[q])
                for key,col in zip(keys,cols,strict=True):
                    p=z['p'][:,col] if isinstance(col,int) else (z[col] if isinstance(col,str) else (z[col[0]]+z[col[1]])/2)
                    out.setdefault(key,np.full(len(ids),np.nan))[m]=p
    with np.load(ART/'exaone_group_20261117/EXAONE_v91_CAST_default8_seed20261112.npz') as z:
        np.testing.assert_array_equal(z['dev_ids'],ids)
        out['EXAONE_grouped']=z['prob']
    for k in range(3):
        with np.load(ART/f'exaone_group_20261117/fold{k}_rows.npz') as z:
            assert np.isin(z['train_ids'],ids).all() and np.isin(z['validation_ids'],ids).all()
            assert not np.isin(g[z['train_ids']],g[z['validation_ids']]).any()
    assert len(out)==10 and all(np.isfinite(v).all() for v in out.values())
    return out


def strata(raw,ids,base):
    side=np.where(base,'positive','negative')
    origin=raw.HomePlanet.iloc[ids].astype('string').fillna('missing').to_numpy(str)
    sleep=raw.CryoSleep.iloc[ids].astype('string').fillna('missing').to_numpy(str)
    region=np.char.add(np.char.add(side,'/'),origin)
    return {'all':np.repeat('all',len(ids)),'direction':side,'direction_origin':region,
            'direction_origin_cryo':np.char.add(np.char.add(region,'/'),sleep)}


def evaluate(hist,full,yn,wrong,discovery,queries,cells,target_cells,names,fold):
    output={}
    for scheme,sc in cells.items():
        qkeys=target_cells[scheme]
        learned={}
        query=np.zeros(len(wrong),bool)
        fitted=np.zeros(len(yn),bool)
        for key in np.union1d(sc[discovery],qkeys):
            t=discovery & (sc==key)
            ct=hist[t].sum(0,dtype=np.int32)
            net=2*hist[t&yn].sum(0,dtype=np.int32)-ct
            j=choose(ct,net,int(t.sum()))
            learned[key]='keep' if j<0 else names[j]
            if j>=0:
                a=qkeys==key
                query[a]=full[a,j]
                fitted[t]=hist[t,j]
        output[scheme]=dict(choices=learned,discovery_fixed=int((fitted&yn).sum()),discovery_harmed=int((fitted&~yn).sum()),
                            evaluations={key:metrics(query&m,wrong,fold) for key,m in queries.items()})
    return output


def run():
    raw,y,g,f,parent,blocks,_,wrong,masks=context()
    panels=sources()[-1]
    h=panels['historical']
    ids=h['ids']; hf=h['fold']
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        np.testing.assert_array_equal(z['rows'],ids)
        np.testing.assert_array_equal(z['y'],y[ids])
        oldref=z['v131']
    older,_,cold=contemporary(ids,oldref,g[ids].astype(int))
    old=older['six']
    common=np.all(np.column_stack(list(older.values()))==old[:,None],axis=1)
    solo=raw.groupby(g).PassengerId.transform('size').eq(1).to_numpy()
    new=~np.isin(g,g[ids])
    assert not np.isin(g[new],g[ids]).any()
    eligible=masks['full']
    queries={'all_new_target':eligible&new,'also_historical_to_new':eligible&new&masks['historical_to_new']}
    ordinary=outcome_scores(h,panel_predictions(h,'historical'))>=.5
    paid=raw[FEES].gt(0).any(axis=1).to_numpy()
    ordinary[~paid[ids],-2:]=old[~paid[ids],None]
    info=archived_outsiders(raw,y,g,ids,hf)
    outside=list(archives(raw,y,g,f,True)[0])
    pos=[i for i,n in enumerate(outside) if n in info]
    others=load_scores(raw,y,g,{'historical':h})['historical'][ids]>=.5
    tabnet=historical_outputs(raw,y,g,h)[0][ids]>=.5
    def label(prefix,each):
        return [prefix+'/'+n for n in each]
    names=label('original_99',[*h['names'],*NEW])+label('other_archives',[outside[i] for i in pos])+label('supplementary',FAMILIES)+label('tabnet_recipes',MODELS)
    hist_bits=np.column_stack([ordinary,*[info[outside[i]]>=.5 for i in pos],others,tabnet])
    votes=np.column_stack([blocks['original_99'],blocks['other_archives'][:,pos],blocks['supplementary'],blocks['tabnet_recipes']])
    assert hist_bits.shape==(len(ids),123) and votes.shape==(len(y),123)
    allids=np.arange(len(y))
    verify_named_pair('historical',ids,hf,raw,y,g,hist_bits,names)
    verify_named_pair('full',allids,f,raw,y,g,votes^parent[:,None],names)
    acts=hist_bits!=old[:,None]
    discovery=solo[ids]&cold&common&acts.any(1)
    # All source labels predate the new-order cohort; the two recent repair gates
    # have no historical-only replay and thus cannot be used to select training rows.
    oh=old!=y[ids]
    keys=strata(raw,ids,old)
    target_keys=strata(raw,allids,parent)
    def counts(data,m):
        return dict(rows=int(m.sum()),errors=int((data&m).sum()))
    r=dict(discovery=counts(oh,discovery),historical_rows=len(ids),
           target_groups={k:counts(wrong,v) for k,v in queries.items()},columns=len(names),matched=names,
           unavailable=['TabDPT_global_compact','LimiX2_fixed512','TabSTAR_finetuned','conditional_bill_density'],
           protocol='Historical source uses only historical labels and solo bookings with ten agreeing replayed releases and an available countervote. Training excludes entire query orders. Historical-only predictions for the two newest repair gates were unavailable, so they apply only in target evaluation. Full target predictions have per-order eightfold exclusion; the entire new cohort was not removed in a single base fit. Nominated two-expert combination was previously selected on all current cases and is retrospective.',
           nominated={})
    for label,columns in [('selected_tree',CHOSEN[:1]),('empty_service',CHOSEN[1:]),('both',CHOSEN)]:
        jj=[names.index(k) for k in columns]
        ht=acts[:,jj].all(1)&discovery
        q=votes[:,jj].all(1)
        r['nominated'][label]=dict(discovery_fixed=int((ht&oh).sum()),discovery_harmed=int((ht&~oh).sum()),
                                  evaluations={k:metrics(q&m,wrong,f) for k,m in queries.items()})
    for tag,start in [('all_matched',0),('non_averaged',18)]:
        ix=np.arange(start,len(names))
        aa=acts[:,ix];bb=votes[:,ix];nn=[names[j] for j in ix]
        ii,jj=np.triu_indices(len(nn),1)
        r[tag]={'single':evaluate(aa,bb,oh,wrong,discovery,queries,keys,target_keys,nn,f),
                'pair':evaluate(aa[:,ii]&aa[:,jj],bb[:,ii]&bb[:,jj],oh,wrong,discovery,queries,keys,target_keys,
                                [nn[l]+' & '+nn[h] for l,h in zip(ii,jj,strict=True)],f)}
    REPORT.write_text(json.dumps(r,indent=2)+'\n')
    print(r['discovery'],r['target_groups'],r['nominated'],flush=True)
    for tag in ['all_matched','non_averaged']:
        for typ,s in r[tag].items():
            print(tag,typ,[(k,x['evaluations']['all_new_target']['net']) for k,x in s.items()],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
