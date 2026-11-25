"""Apply the marginal change in estimated odds from exact amounts to the incumbent."""
import argparse
import json
import numpy as np
from scipy.special import expit,logit

from full_roster_booking_validation_n23 import ROOT,data

BASE=ROOT/'artifacts/nov25_passenger_joint_likelihood'
OUT=ROOT/'artifacts/nov25_within_bin_bill_precision'


def replay():
    import torch
    from passenger_joint_likelihood_n25 import encode,train_model,TARGET
    from within_bin_bill_precision_n25 import coordinates,predictions
    torch.set_num_threads(2)
    raw,y,g,f=data()
    for k in range(8):
        q,t=np.flatnonzero(f==k),np.flatnonzero(f!=k)
        encoded,cards=encode(raw,t)
        x=torch.as_tensor(np.column_stack([encoded,y.astype(int)]))
        query=x[q].clone()
        query[:,TARGET]=0
        pos,shuf=coordinates(raw,encoded,t,q,k)
        path=OUT/f'{k}_query_intervention.npz'
        if not path.exists():
            values={}
            for mode,coords in [('exact_bill_position',pos),('shuffled_bill_position',shuf)]:
                model=train_model(x,cards,t,1241+k,True,coords)
                p=predictions(model,query,coords[q])
                with np.load(OUT/f'{k}_{mode}.npz') as z:
                    for n in p:np.testing.assert_array_equal(z[n],p[n])
                for condition,test_pos in [('none',torch.zeros_like(pos[q])),('observed',pos[q]),('permuted',shuf[q])]:
                    with torch.inference_mode():
                        mask=torch.ones_like(query,dtype=torch.bool)
                        mask[:,TARGET]=False
                        values[mode+'_'+condition]=model(query,mask,test_pos)[:,TARGET,:2].softmax(1)[:,1].numpy()
            with path.with_suffix('.tmp').open('wb') as s:
                np.savez_compressed(s,q=q,t=t,labels=y[t],**values)
            path.with_suffix('.tmp').replace(path)
        print('verified',k,flush=True)


def evaluate():
    from disjoint_tree_fields_n25 import ART,DEST,stats
    from full_roster_six_failure_archive_n24 import prediction_sets
    from grouped_peer_release_replay_n25 import edit_effect_bounds
    from grouped_peer_teacher_n25 import OUT as REF
    raw,y,g,f=data()
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        rows,hf=z['rows'],z['fold']
    _,panels=prediction_sets(raw,y,g,f,rows,hf)
    votes=(np.column_stack(panels[0])>=.5).sum(1)
    majority=votes>34
    common=(votes==0)|(votes==68)
    def odds(p):return logit(p.clip(.002,.998))
    values={}
    all_q=[]
    known=[]
    for k in range(8):
        q=np.flatnonzero(f==k)
        all_q.append(q)
        with np.load(BASE/f'fold{k}.npz') as z:base_model=odds(z['direct'])
        with np.load(OUT/f'{k}_query_intervention.npz') as z:
            np.testing.assert_array_equal(z['q'],q)
            delta={}
            for key in ('exact_bill_position','shuffled_bill_position'):
                observed,none,permuted=[odds(z[key+'_'+tag]) for tag in ('observed','none','permuted')]
                delta[key+'_training_and_query']=(permuted if key.startswith('shuffled') else observed)-base_model
                delta[key+'_query_only']=observed-none
                delta[key+'_placebo_query']=permuted-none
        with np.load(REF/f'release{k}.npz') as z:
            native,prior,lo,hi=(z[key] for key in ('native','v131','nine_on_low_branch','nine_on_high_branch'))
        known.append(prior>=0)
        for name,change in delta.items():
            for scale in (.5,1.):
                pred=odds(native)+scale*change>=0
                a,b=edit_effect_bounds(pred,native>=.5,prior,lo,hi,y[q])
                d=((pred==y[q]).astype(int)-(majority[q]==y[q]))
                values.setdefault(name+'/'+str(scale),[]).append((a,b,d))
    q=np.concatenate(all_q)
    known=np.concatenate(known)
    r={}
    for name,blocks in values.items():
        a,b,d=(np.concatenate([v[j] for v in blocks]) for j in range(3))
        np.testing.assert_array_equal(a[known],b[known])
        r[name]=dict(bounds=[int(a.sum()),int(b.sum())],known=stats(a,known,f[q],g[q]),
                     new_order=stats(a,known&~np.isin(g[q],g[rows]),f[q],g[q]),
                     common=stats(d,common[q],f[q],g[q]))
        out=r[name]
        print(name,out['bounds'],out['known']['fixed'],out['known']['harmed'],out['common']['fixed'],out['common']['harmed'])
    DEST.with_name('transfer_bill_precision.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['replay','evaluate'])
    replay() if p.parse_args().action=='replay' else evaluate()
