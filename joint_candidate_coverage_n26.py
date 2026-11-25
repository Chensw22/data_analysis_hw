"""Measure overlaps and marginal accuracy of frozen candidate decisions on the same passenger and parent branch."""
import json
import numpy as np
import pandas as pd

from awake_context_ranks_n25 import OUT as NEURAL
from awake_error_demonstrations_n25 import OUT as ERRORS
from disjoint_tree_fields_n25 import ART,DEST,OUT as VIEWS,stats
from dispute_band_stability_n26 import gate
from family_identity_transfer_n26 import panels_and_reference
from full_roster_booking_validation_n23 import ROOT,data
from kaggle_workflow import sha256
from scipy.special import expit,logit


def full_inputs(k,q,native,awake):
    candidates=np.zeros((3,len(q)),bool)
    a=awake[q]
    with np.load(NEURAL/f'{k}_personal.npz') as z:
        np.testing.assert_array_equal(z['q'],q[a])
        candidates[0,a]=(z['p']>=.5)!=(native[a]>=.5)
    with np.load(ERRORS/f'{k}_balanced_error.npz') as z:
        np.testing.assert_array_equal(z['q'],q[a])
        side=native[a]>=.5
        fraction=z['training_correct_fraction'][side.astype(int)]
        prob=expit(logit(z['p'].clip(.00001,.99999))+np.log(fraction))
        candidates[1,a]=candidates[0,a]&((np.where(side,1-prob,prob)>=.5)!=side)
    return candidates[:2]


def branch_decisions(row,fold,panel,names,native,prior,lo,hi,awake,tag,seed):
    trees=[i for i,name in enumerate(names) if name.startswith('permutation_selection/')]
    assert len(trees)==5
    band=gate(panel)
    tree=panel[:,trees].mean(1)>=.5
    change=(.75*native+.25*panel[:,names.index('permutation_selection/random')]>=.5)!=(native>=.5)
    net=np.zeros(len(row),bool)
    filt=net.copy()
    for k in np.unique(fold):
        m=fold==k
        q=row[m]
        confirmed=change[m].copy()
        for suffix in ('','_repeat'):
            with np.load(VIEWS/f'{tag}_{k}{suffix}.npz') as z:
                np.testing.assert_array_equal(z['q'],q)
                confirmed &= ((z['p']>=.5)!=(native[m,None]>=.5)).sum(1)>=2
        change[m]=confirmed
        if tag=='full':
            net[m],filt[m]=full_inputs(k,q,native[m],awake)
        else:
            a=awake[q]
            with np.load(ART/f'nov26_historical_awake/{k}_{seed+k}.npz') as z:
                np.testing.assert_array_equal(z['q'],q[a])
                net[np.flatnonzero(m)[a]]=(z['p']>=.5)!=(native[m][a]>=.5)
    out={}
    for side,branch in [(False,lo),(True,hi)]:
        parent=np.where(prior>=0,prior,side).astype(bool)
        nine=np.where(prior>=0,np.where(parent,hi,lo),branch).astype(bool)
        eligible=(parent==(native>=.5))&(nine==parent)
        b=band&(tree!=nine)
        v=(net|change)&eligible
        masks={'nine':np.zeros(len(b),bool),'band':b,'fields':change&eligible,'awake':net&eligible,
               'fields_awake':v,'all_three':v|b,'band_only':b&~v,'fields_awake_only':v&~b,'overlap':v&b}
        if tag=='full':
            masks.update(filtered=filt&eligible,band_and_filtered=b|(filt&eligible))
        for key,flip in masks.items():
            out.setdefault(key,[]).append(nine^flip)
    return {k:np.stack(pred) for k,pred in out.items()}


def run():
    raw,y,g,f=data()
    _,_,_,rows,names,p,h,native,old_native,reference=panels_and_reference()
    awake=raw.CryoSleep.eq(False).to_numpy()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old_nine=z['nine']; old_prior=z['v131'];hf=z['fold']
    fresh=~np.isin(g,g[rows])
    cold=np.zeros(len(rows),bool)
    for k in np.unique(hf):
        cold[hf==k]=~np.isin(g[rows[hf==k]],g[rows[hf!=k]])
    report={}
    for tag,row,fold,panel,base,prior,lo,hi,seed in [
        ('full',np.arange(len(y)),f,p,native,*reference,891),
        ('historical',rows,hf,h,old_native,old_prior,old_nine,old_nine,891),
        ('historical_alternate_seed',rows,hf,h,old_native,old_prior,old_nine,old_nine,2991)]:
        is_full=tag=='full'
        pred=branch_decisions(row,fold,panel,names,base,prior,lo,hi,awake,'full' if is_full else 'historical',seed)
        votes=(panel>=.5).sum(1)
        known=prior>=0
        cohorts={'all':np.ones(len(row),bool),'unanimous':(votes==0)|(votes==68)}
        if is_full:
            cohorts['new_orders']=fresh
        else:
            cohorts['complete_order_holdout']=cold
        metrics={}
        for key,comp in [(key,'nine') for key in pred if key!='nine']+[
                ('all_three','band'),('all_three','fields_awake'),('all_three','awake')]:
            a=(pred[key]==y[row]).astype(int)-(pred[comp]==y[row])
            np.testing.assert_array_equal(a[0,known],a[1,known])
            result={}
            for title,mask in cohorts.items():
                r=stats(a[0],mask&known,fold,g[row])
                r['whole_bounds']=[int(fun(a,axis=0)[mask].sum()) for fun in (np.min,np.max)]
                result[title]=r
            metrics[key+'_vs_'+comp]=result
        report[tag]=metrics
        for key in ['band_vs_nine','fields_awake_vs_nine','all_three_vs_nine','all_three_vs_band','all_three_vs_fields_awake']:
            entry=metrics[key]
            print(tag,key,[(title,v['fixed'],v['harmed'],v['whole_bounds']) for title,v in entry.items()],flush=True)
    original=json.loads(DEST.with_name('confirmed_awake_union.json').read_text())['union']['all']['known']
    current=report['full']['fields_awake_vs_nine']['all']
    assert (current['fixed'],current['harmed'])==(original['fixed'],original['harmed'])
    for tag,n in [('full',(80,65)),('historical',(90,81))]:
        r=report[tag]['band_vs_nine']['all']
        assert (r['fixed'],r['harmed'])==n
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    masks={}
    for name,folder in [('nine','n24_selected_dissent'),('band','n26_dispute_band'),('fields_awake','n25_confirmed_awake_union'),
                         ('filtered','n25_awake_error_confirmation')]:
        path=ART/('proposal_'+folder)/'submission.csv'
        assert sha256(path)==json.loads((ROOT/'reports'/('proposal_'+folder)/'manifest.json').read_text())['sha256']
        frame=pd.read_csv(path)
        np.testing.assert_array_equal(frame.PassengerId,test.PassengerId)
        masks[name]=frame.Transported.to_numpy(bool)
    band=masks['band']!=masks['nine'];union=masks['fields_awake']!=masks['nine']; filtered=masks['filtered']!=masks['nine']
    report['competition']=dict(band=int(band.sum()),fields_awake=int(union.sum()),overlap=int((band&union).sum()),
        all_three=int((band|union).sum()),band_only=int((band&~union).sum()),fields_awake_only=int((union&~band).sum()),
        band_and_filtered=int((band|filtered).sum()))
    DEST.with_name('joint_candidate_coverage.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    run()
