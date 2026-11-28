"""Apply both occlusion policies with matched training exposure, including for test passengers."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits
from block_response_curvature_n28 import (ART,ROOT,OUT,inputs,cohort,count,source,PARENTS,
    design,response,calibrated,validate_submission,sha256,write)

NAME='proposal_n28_repeated_block_response'


def policy(margin,parent,score):
    return ((score[:,1]>=0)!=parent)&((margin+score[:,1]-score[:,0]>=0)!=parent)


def tokens(strings,salt):
    return np.array([int.from_bytes(hashlib.sha256((salt+v).encode()).digest()[:4],'big') for v in strings],dtype=np.int64)


def run():
    frame,y,g,f,ids,_,_,seven,versions=inputs();n=len(y);test=np.arange(n,len(frame))
    parent=versions['random'][:,0]
    _,_,known,fresh=cohort(frame.iloc[:n],y,f)
    soft,official,digest=source(PARENTS['direction'],frame.iloc[n:])
    np.testing.assert_array_equal(parent[n:],official)
    for j,name in enumerate(PARENTS.values()):
        _,d,_=source(name,frame.iloc[n:])
        for v in versions.values():np.testing.assert_array_equal(v[n:,j],d)
    margin=np.where(parent,1.,-1.)*np.maximum(.00001,abs(logit(np.median(seven[:,1:4],axis=1).clip(.003,.997))))
    x,blocks=design(frame); tg=frame.PassengerId.to_numpy(str)[n:].astype(str)
    tg=np.array([s[:4] for s in tg])
    directions={};scores={};part=np.zeros(n,int)
    for k,chunk in enumerate(np.array_split(np.unique(g),8)):part[np.isin(g,chunk)]=k
    with np.load(OUT/'aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids);np.testing.assert_array_equal(z['y'],y)
        local={k:z[k] for k in z.files if k.endswith('_flag') or k.endswith('_scores')}
    for tag,fold in [('random',f),('ordered',part)]:
        if tag=='random': assign=tokens(tg,'nov28-response-outer')%8
        else: assign=np.searchsorted([g[part==i].max() for i in range(7)],tg,side='left')
        uv=np.full((len(test),2),np.nan)
        for k in range(8):
            candidates=np.flatnonzero(assign==k)
            if len(candidates)==0:continue
            t=np.flatnonzero(fold!=k);held=np.flatnonzero(fold==k)
            q=np.r_[held,test[candidates]]
            scored=response(x,blocks,t,q,y)
            with np.load(OUT/f'{tag}_{k}.npz') as c:np.testing.assert_allclose(c['response'],scored[:len(held)])
            train=np.arange(len(held)); predict=np.arange(len(held),len(q))
            orders=np.unique(g[held])
            if tag=='random':orders=np.random.default_rng(726+k).permutation(orders)
            bunches=np.array_split(orders,4)
            if tag=='random': positions=tokens(tg[candidates],'nov28-response-inner')%4
            else: positions=np.searchsorted([a.max() for a in bunches[:-1]],tg[candidates],side='left')
            for l,bunch in enumerate(bunches):
                use=train[~np.isin(g[held],bunch)]
                query=predict[positions==l]
                if not len(query):continue
                for j,m in enumerate(['plain','sensitivity']):
                    uv[candidates[positions==l],j]=calibrated(scored,use,query,y[held],None,m,k*13+l)
        assert np.isfinite(uv).all()
        scores[tag]=uv
        directions[tag]=np.r_[local[tag+'_flag'][:n],policy(margin[n:],official,uv)]
    vote=directions['random']&directions['ordered']
    wrong=parent[:n]!=y
    same7=((seven>=.5)==parent[:,None]).all(1)
    same4=np.logical_and.reduce([(p==parent[:,None]).all(axis=1) for p in versions.values()])
    same= same4 & same7
    report=dict(known_rows=int(known.sum()), accuracy=float((y[known]==(parent[:n]^vote[:n])[known]).mean()),
        cohorts={name:count(vote[:n],m,wrong) for name,m in dict(all=known,fresh=known&fresh,
                    four=known&same4[:n],seven=known&same7[:n],four_and_seven=known&same[:n],other=known&~same[:n]).items()},
        test=int(vote[n:].sum()),test_four_agree=int((vote[n:]&same4[n:]).sum()),test_four_and_seven=int((vote[n:]&same[n:]).sum()),
        test_protocol_counts={k:int(v[n:].sum()) for k,v in directions.items()},
        fold_net=[int((vote[:n]*np.where(wrong,1,-1))[known&(f==k)].sum()) for k in range(8)])
    nulls=[]
    for k in ('random','ordered'):
        arr=local[k+'_scores']
        nulls.append(((arr[:n,2]>=0)!=parent[:n])&((margin[:n]+arr[:n,2]-arr[:n,0]>=0)!=parent[:n]))
    report['shuffled_both']=count(nulls[0]&nulls[1],known,wrong)
    out=ART/NAME;out.mkdir(exist_ok=True);model=out/'inputs.npz'
    np.savez_compressed(model,passenger=ids[n:],parent=soft,margin=margin[n:],**scores)
    with np.load(model) as z:
        a=policy(z['margin'],z['parent']>=.5,z['random'])&policy(z['margin'],z['parent']>=.5,z['ordered'])
        p=np.where(a,1.-z['parent'],z['parent']);p[(z['parent']==.5)&a]=.499999
    np.testing.assert_array_equal(p>=.5,official^vote[n:])
    csv=out/'submission.csv'
    pd.DataFrame(dict(PassengerId=ids[n:],Transported=p>=.5)).to_csv(csv,index=False)
    pd.DataFrame(dict(PassengerId=ids[n:],probability=p,reload_probability=p)).to_csv(out/'test_probabilities.csv',index=False)
    check=validate_submission(csv,ROOT/'data/raw/test.csv')
    others=[]
    for path in (ROOT/'reports').glob('*/online_attempts/*.json'):
        r=json.loads(path.read_text());src=ART/path.parents[1].name/'submission.csv'
        if r.get('evaluation',{}).get('Status')=='COMPLETE' and src.exists() and sha256(src)==r['file_sha256']:
            b=pd.read_csv(src);np.testing.assert_array_equal(b.PassengerId,ids[n:])
            others.append([int(np.sum((p>=.5)!=b.Transported)),path.parents[1].name])
    report['nearest']=min(others); report['orders']=int(np.unique(tg[a]).size)
    assert report['nearest'][0]>0
    version=NAME.removeprefix('proposal_')
    for loc in (out,ROOT/'reports'/NAME):
        write(loc/'manifest.json',dict(**check,version=version,model_sha256=sha256(model),parents={PARENTS['direction']:digest},
             reload_difference=0,training_rows=n,changed=int(a.sum())))
        write(loc/'selection.json',dict(version=version,rule='Two independently trained booking protocols must support the incremental pair-mask response. Test receives held-training-size models and readouts.',evidence='reports/nov23_matched_learning_curve/matched_block_response_export_n28.json'))
    write(ROOT/'reports/nov23_matched_learning_curve/matched_block_response_export_n28.json',report);print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(2):run()
