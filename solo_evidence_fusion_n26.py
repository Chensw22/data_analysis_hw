"""Compare four new sources of solo evidence on the errors no archived model reached."""
import json

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from disjoint_tree_fields_n25 import ART, DEST
from family_identity_transfer_n26 import fit
from manifest_role_joint_n26 import roles
from solo_archive_recovery_n26 import archive, ticker
from solo_selection_likelihood_n26 import regimes


def odds(prob):
    return logit(np.clip(prob,.002,.998))


def evidence(tag, k, q, age_known, anchor):
    def load(folder, keys):
        with np.load(ART / ('nov26_'+folder) / f'{tag}_{k}.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            return [z[key] for key in keys]
    prior, direct, a, b = load('solo_selection_likelihood',['blind','direct','true_0','true_1'])
    size = odds(prior)+np.log(b/a)
    role_direct, table = load('manifest_role_joint',['direct','real'])
    # The caller evaluates solo passengers only.
    conditioned = odds(table[:,0,1]/table[:,0].sum(1))
    band, age_direct, year = load('conditional_age_density',['prior','direct','true_year'])
    age = np.where(age_known, odds(band)+np.log(year[:,1]/year[:,0]), odds(age_direct))
    witness, count = load('pareto_bill_witnesses',['scores','counts'])
    index = np.where(anchor,2,1)
    i = np.arange(len(q))
    ordinary_witness = odds(witness[:,0])
    ordinary_witness[count[:,0]<8]=0.
    bound = np.where(count[i,index]>=8,odds(witness[i,index]),ordinary_witness)
    pre=np.column_stack([odds(direct),odds(role_direct),odds(age_direct),ordinary_witness])
    post=np.column_stack([size,conditioned,age,bound])
    # Lack of an eligible witness is an abstention, independently of anchor direction.
    support=np.ones_like(post,dtype=bool)
    support[:,3]=count[i,index]>=8
    return pre,post,support


def rank(score, wrong, ids):
    if not len(ids) or len(np.unique(wrong[ids]))!=2:
        return {}
    chosen=ids[np.argsort(-score[ids],kind='stable')[:int(np.ceil(len(ids)*.1))]]
    return dict(rows=len(ids), errors=int(wrong[ids].sum()), auc=float(roc_auc_score(wrong[ids],score[ids])),
                top_tenth_rows=len(chosen), top_tenth_errors=int(wrong[chosen].sum()))


def run():
    raw,y,g,views=regimes()
    alone=roles(raw,g)==0
    coefficients={}
    reports={}
    for tag,ids,fold,native,view in views:
        anchor=view['anchor']
        wrong=anchor!=y[ids]
        sign=np.where(anchor,-1,1)
        eligible=view['known']&alone[ids]
        common=view['eligible']&(view['votes']==anchor[:,None]).all(1)&alone[ids]
        _,old_votes=archive(tag,ids,fold,g,y,ART/'nov24_remaining_cases/archive_scores.npz')
        hard=common&(old_votes==anchor[:,None]).all(1)
        pre=np.empty((len(ids),4));post=np.empty_like(pre);support=np.empty_like(pre,dtype=bool)
        for k in np.unique(fold):
            mask=fold==k
            pre[mask],post[mask],support[mask]=evidence(tag,k,ids[mask],raw.Age.iloc[ids[mask]].notna(),anchor[mask])
        opposed=(post*sign[:,None]>0)&support
        new=opposed&(pre*sign[:,None]<=0)
        scopes={'solo':eligible,'common':common,'unreached':hard,'unreached_negative':hard&~anchor}
        offset=sign*odds(native)
        ordinary=sign[:,None]*pre-offset[:,None]
        changes=sign[:,None]*(post-pre)*support
        mat=np.column_stack([np.ones(len(ids)),sign,offset,ordinary])
        designs={'ordinary':mat,'with_evidence':np.column_stack([mat,changes,changes*sign[:,None]])}
        actions={'any':opposed.any(1),'any_induced':new.any(1),'two':opposed.sum(1)>=2,
                 'two_with_induced':(opposed.sum(1)>=2)&new.any(1)}
        scores={}
        for set_name,trainset in [('solo',eligible),('unreached',hard)]:
            for name,x in designs.items():
                key=set_name+'_'+name
                local_score=offset.copy()
                for k in np.unique(fold):
                    pool=np.flatnonzero((fold==k)&trainset)
                    for t,q in KFold(3,shuffle=True,random_state=402).split(pool):
                        t,q=pool[t],pool[q]
                        assert not np.isin(g[ids[t]],g[ids[q]]).any()
                        local_score[q]=offset[q]+x[q]@fit(x[t],offset[t],wrong[t])
                actions['local_'+key]=(local_score>0)&trainset
                scores['local_'+key]=local_score
                if tag=='historical':
                    coefficients[key]=fit(x[trainset],offset[trainset],wrong[trainset])
                else:
                    fresh=~np.isin(g[ids],g[views[0][1]])
                    transferred=offset+x@coefficients[key]
                    actions['transfer_'+key]=(transferred>0)&fresh&trainset
                    scores['transfer_'+key]=transferred
        report={}
        for name,flag in actions.items():
            subset=scopes if not name.startswith('transfer_') else {n:mask&fresh for n,mask in scopes.items()}
            report[name]={c:ticker(mask,wrong,flag) for c,mask in subset.items()}
        if tag=='full':
            assert report['any']['unreached']['errors']==179
            scope=fresh&hard&~anchor
            report['ranking_new_unreached_negative']={k:rank(s,wrong,np.flatnonzero(scope)) for k,s in
                [('native',offset),*[(k,v) for k,v in scores.items() if k.startswith('transfer_')]]}
        reports[tag]=report
        print(tag,[(k,v['unreached']['fixed'],v['unreached']['harmed']) for k,v in report.items() if 'unreached' in v],flush=True)
    DEST.with_name('solo_evidence_fusion.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(2):
        run()
