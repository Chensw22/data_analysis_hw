"""Validate the frozen low-margin review budget among unanimous awake paying/sleeping unpaid passengers."""
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from full_roster_booking_validation_n23 import ROOT,data
from price_implied_deck_n26 import FEES
from repeated_archive_misses_n26 import ART,cohort
from six_way_against_scored_nine_n26 import test_decisions

REPORT=ROOT/'reports/nov23_matched_learning_curve/low_margin_enrichment_audit.json'
OUT=ART/'nov27_low_margin_enrichment_audit'
PARENT='proposal_n27_directional_foundation'
SHARE=.25


def within_rank(margin,groups,fold,eligible):
    result=np.zeros(len(margin),bool)
    quart=np.full(len(margin),-1,int)
    for cell in np.unique(groups[eligible]):
        for k in np.unique(fold[eligible]):
            q=np.flatnonzero(eligible&(groups==cell)&(fold==k))
            if len(q):
                frac=rankdata(margin[q],method='average')/len(q)
                result[q]=frac<=SHARE;quart[q]=np.ceil(4*frac).astype(int)-1
    return result,quart


def external_cut(v):
    # Reproduce average-rank <= 25% on a reference distribution; release a single numeric cutoff.
    frac=rankdata(v,method='average')/len(v)
    return v[frac<=SHARE].max(initial=-np.inf)


def reference_selection(margin,state,fold,parts,groups,eligible,bankmask,querymask):
    selected=np.zeros(len(margin),bool)
    for k in np.unique(fold):
        for j in np.unique(parts[querymask]):
            for st in [0,1]:
                q=np.flatnonzero(eligible&querymask&(fold==k)&(parts==j)&(state==st))
                t=np.flatnonzero(eligible&bankmask&(fold==k)&(parts!=j)&(state==st))
                if not len(q):continue
                assert len(t) and not np.isin(groups[t],groups[q]).any()
                selected[q]=margin[q]<=external_cut(margin[t])
    return selected


def metrics(eligible,sel,err):
    r=eligible&sel;u=eligible&~sel
    def ratio(a,b):return float(a/b) if b else None
    pr=ratio(err[r].sum(),r.sum());pu=ratio(err[u].sum(),u.sum())
    return dict(eligible=int(eligible.sum()),errors=int(err[eligible].sum()),selected=int(r.sum()),selected_errors=int(err[r].sum()),
          other=int(u.sum()),other_errors=int(err[u].sum()),rate_selected=pr,rate_other=pu,
          risk_difference=None if pr is None or pu is None else pr-pu,recall=ratio(err[r].sum(),err[eligible].sum()),
          relative_risk=ratio(pr,pu) if pr is not None and pu is not None else None)


def intervals(eligible,select,wrong,group,strata,reps=6000):
    index=np.flatnonzero(eligible)
    _,gid=np.unique(group[index],return_inverse=True)
    _,cells=np.unique(strata[index],return_inverse=True)
    p=len(np.unique(cells));book=np.zeros((gid.max()+1,p,2,2),np.float32)
    np.add.at(book,(gid,cells,select[index].astype(int),wrong[index].astype(int)),1)
    rng=np.random.default_rng(512729)
    sim=[];unsupported=[]
    for lo in range(0,reps,128):
        w=rng.exponential(size=(min(reps-lo,128),len(book))).astype(np.float32)
        a=(w@book.reshape(len(book),-1)).reshape(len(w),p,2,2)
        sizes=a.sum(3)
        rate=np.divide(a[:,:,:,1],sizes,out=np.zeros_like(sizes),where=sizes>0)
        supported=sizes[:,:,0]>0
        mass=sizes[:,:,1]*supported
        standardized=(mass*(rate[:,:,1]-rate[:,:,0])).sum(1)/mass.sum(1)
        all_counts=a.sum(1)
        unadj=all_counts[:,1,1]/all_counts[:,1,:].sum(1)-all_counts[:,0,1]/all_counts[:,0,:].sum(1)
        sim.append(np.column_stack([unadj,standardized]))
        unsupported.append(np.sum(sizes[:,:,1]*~supported,axis=1)/sizes[:,:,1].sum(1))
    samples=np.vstack(sim)
    a=book.sum(0);size=a.sum(2);valid=(size[:,0]>0)
    rates=np.divide(a[:,:,1],size,out=np.zeros_like(size),where=size>0)
    mass=size[:,1]*valid
    adj=float((mass*(rates[:,1]-rates[:,0])).sum()/mass.sum())
    return dict(orders=int(len(book)),unadjusted_95=np.quantile(samples[:,0],[.025,.975]).tolist(),
          adjusted_difference=adj,adjusted_95=np.quantile(samples[:,1],[.025,.975]).tolist(),
          selected_without_comparison=int((size[:,1]*~valid).sum()),cells=p,
          standardization='Within behavior, frozen parent decision and solo/party; weighted to the selected passenger composition using supported cells.')


def run():
    raw,y,g,fold=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns);frame=pd.concat([raw,test],ignore_index=True)
    ids=frame.PassengerId.to_numpy(str);_,blind,known,fresh=cohort(raw,y,fold)
    with np.load(ART/'nov25_historical_nine/historical.npz') as z:
        old=z['rows'];np.testing.assert_array_equal(z['passenger'],ids[old])
    np.testing.assert_array_equal(fresh,~np.isin(g,g[old]))
    with np.load(ART/'nov27_dissent_likelihood_transfer/aligned.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids);first=z['first'];anchor=first^z['random_median']
    np.testing.assert_array_equal(test_decisions(PARENT,test.PassengerId)[0],anchor[n:])
    with np.load(ART/'nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['parent'],anchor)
        scores=z['opinions']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        foundation=z['scores'][:,-3:];past=np.median(z['scores'][:,:18],axis=1)
    with np.load(ART/'nov27_outsider_directional/aligned.npz') as z:outside=[z['dpt'],np.median(z['tabnet'],axis=1)]
    np.testing.assert_array_equal(scores,np.column_stack([past,foundation,np.median(foundation,axis=1),*outside]))
    common=((scores>=.5)==anchor[:,None]).all(1)
    vote_common=((scores>=.5)==(scores[:,:1]>=.5)).all(1)
    fees=frame[FEES]
    paid=fees.gt(0).any(axis=1).to_numpy();missing=fees.isna().any(axis=1).to_numpy()
    awake=frame.CryoSleep.eq(False).to_numpy()&paid
    sleepy=frame.CryoSleep.eq(True).to_numpy()&~paid
    state=np.select([awake,sleepy],[0,1],default=-1)
    allg=frame.PassengerId.str.split('_').str[0].to_numpy()
    party=pd.Series(allg).duplicated(keep=False).to_numpy()
    eligible=np.r_[known,np.ones(len(test),bool)]&common&(state>=0)
    margin=np.min(abs(scores-.5),axis=1)
    partition=np.r_[fold,np.full(len(test),8)]
    original,quartile=within_rank(margin,state,partition,eligible)
    known_err=anchor[:n]!=y
    reference=json.loads((ROOT/'reports/nov23_matched_learning_curve/common_error_review_coverage.json').read_text())
    for k,name in enumerate(['awake_paid','sleep_zero']):
        m=eligible&(state==k)
        row=reference[name]['0.25'];allrow=reference[name]['1.0']
        assert (int(m[:n].sum()),int((m[:n]&known_err).sum()))==(allrow['reviewed'],allrow['errors'])
        assert (int((m[:n]&original[:n]).sum()),int((m[:n]&original[:n]&known_err).sum()),int((m[n:]&original[n:]).sum()))==(row['reviewed'],row['errors'],row['test_reviewed'])
    assert original[:n].sum()==1623 and (original[:n]&known_err).sum()==481
    base=anchor[:n];eligible_train=eligible[:n];st=state[:n];m=margin[:n]
    unique=np.unique(g)
    ordered=np.zeros(n,int);random=np.zeros(n,int)
    for k,book in enumerate(np.array_split(unique,8)):ordered[np.isin(g,book)]=k
    for k,book in enumerate(np.array_split(np.random.default_rng(17171).permutation(unique),8)):random[np.isin(g,book)]=k
    select={'published':original[:n]}
    for name,parts in [('held_random',random),('held_ordered',ordered)]:
        select[name]=reference_selection(m,st,fold,parts,g,eligible_train,known,known)
    select['historic_to_fresh']=reference_selection(m,st,fold,fresh.astype(int),g,eligible_train,known&~fresh,known&fresh)
    rr={'parent':PARENT,'published_verified':True}
    def ledger(gate):
        return dict(rows=int(gate.sum()),errors=int((gate&known_err).sum()))
    usable=known&common[:n]
    rr['denominators']={'all_train':len(y),'known':ledger(known),'all_seven_match_parent':ledger(usable),
        'two_state':ledger(eligible_train),'other_behavior':ledger(usable&(st<0)),'31_way_unanimous_and_two_state':ledger(blind&eligible_train),
        'self_unanimous_but_opposes_parent':ledger(known&vote_common[:n]&~common[:n]),
        'self_unanimous_but_opposes_parent_in_two_states':ledger(known&vote_common[:n]&~common[:n]&(st>=0)),
        'seven_plus_111_agreement':ledger(usable&(first[:n]==base)),
        'historically_unseen_rows':int((fresh&known).sum()),'unknown_rows_entire_train':int((~known).sum())}
    counterpart=np.r_[known,np.ones(len(test),bool)]&vote_common&(state>=0)
    unanchored,_=within_rank(margin,state,partition,counterpart)
    rr['denominators']['if_seven_agree_to_each_other_only']={
        'two_state':ledger(counterpart[:n]),'reviewed':ledger(unanchored[:n]),
        'newly_reviewed':ledger(unanchored[:n]&~original[:n]),'dropped':ledger(original[:n]&~unanchored[:n]),
        'test_reviewed':int(unanchored[n:].sum())}
    rr['test_review_coverage']={key:dict(eligible=int((eligible[n:]&(state[n:]==k)).sum()),selected=int((original[n:]&(state[n:]==k)).sum()))
                                for k,key in enumerate(['awake_paid','sleep_zero'])}
    rr['missingness']={name:metrics(eligible_train&(st==k)&(missing[:n]==miss),original[:n],known_err)
        for k,tag in [(0,'awake'),(1,'sleep')] for miss,name in [(False,tag+'_complete'),(True,tag+'_partial')]}
    strata=st*4+base.astype(int)*2+party[:n].astype(int)
    results={}
    for name,sel in select.items():
        allowed=eligible_train&(fresh if name=='historic_to_fresh' else np.ones(n,bool))
        scopes=dict(all=allowed,awake=allowed&(st==0),sleep=allowed&(st==1),known_historical_orders=allowed&~fresh,
                    historical_outside=allowed&fresh,solo=allowed&~party[:n],party=allowed&party[:n])
        for k,label in [(0,'awake'),(1,'sleep')]:
            for side in [False,True]:
                scopes[label+'_parent_'+str(int(side))]=allowed&(st==k)&(base==side)
        res={k:metrics(v,sel,known_err) for k,v in scopes.items()}
        res['order_ci']={k:intervals(v,sel,known_err,g,strata) for k,v in scopes.items() if k in ['all','historical_outside','awake','sleep'] and v.any()}
        res['per_original_fold']=[metrics(allowed&(fold==i),sel,known_err) for i in range(8)]
        res['per_ordered_block']=[metrics(allowed&(ordered==i),sel,known_err) for i in range(8)]
        res['per_random_block']=[metrics(allowed&(random==i),sel,known_err) for i in range(8)]
        res['overlap_with_published']=int((allowed&sel&original[:n]).sum())
        results[name]=res
    rr['protocols']=results
    rr['quartiles']={}
    for v,scope in [('all',eligible_train),('awake',eligible_train&(st==0)),('sleep',eligible_train&(st==1)),('fresh',eligible_train&fresh)]:
        rr['quartiles'][v]=dict(cells=[ledger(scope&(quartile[:n]==j)) for j in range(4)],ranking_auc=float(roc_auc_score(known_err[scope],-m[scope])))
    rr['semantics']=['All rates use the registered 16-change parent and the same seven held-order scores. Two are correlated composites, including the median of the three foundation scores, so votes are not independent.',
        'In the published rule, percentile ranks are computed within each of the eight held folds and each of the two behaviors with ties assigned average ranks. Fixed-margin reference checks estimate a threshold using different whole bookings within the original held fold; each reference/query pair was unseen by the same underlying models.',
        'Random/contiguous blocks are pseudo holdouts for the margin selection only. Their scores, trained original model and parent are shared, and booking number is an ordering proxy. No full ensemble was retrained on contiguous outer regions.',
        'The historical-outside mask excludes every booking represented in the original historical cache, but the aggregate results of this subpopulation have been consulted in prior project iterations. This is not a pristine final evaluation.',
        'Two behavior groups and the 25% review budget were inspected in prior work. The intervals resample whole bookings conditional on that choice; they do not account for exploratory selection.',
        'No-positive-spend includes partially missing bills, as in the source. Error enrichment measures the precision of review; reversing all reviewed predictions is a distinct decision.']
    OUT.mkdir(exist_ok=True);np.savez_compressed(OUT/'masks.npz',passenger=ids[:n],fold=fold,ordered=ordered,random=random,y=y,parent=base,
             known=known,fresh=fresh,state=st,eligible=eligible_train,margin=m,quartile=quartile[:n],**select)
    REPORT.write_text(json.dumps(rr,indent=2)+'\n')
    fmt=lambda x:f'{100*x:.1f}%'
    summary=results['published']['all']
    lines=['# 七路同判低置信度乘客的错例富集', '',
           f"以公榜82.370%的16人版为同一父版，在两类七票同判的{summary['eligible']:,}名乘客中，最低置信度四分之一的错误率为{fmt(summary['rate_selected'])}，其余乘客为{fmt(summary['rate_other'])}。该规则覆盖这两类{summary['errors']}个错例中的481个，占所有可核验父版1,435个错例的33.5%。将这1,623人全部改判会改对481人、改错1,142人。", '',
           '| 检验 | 被选错误 / 人数 | 未选错误 / 人数 | 被选错误率 | 未选错误率 | 两类错例召回 |',
           '|---|---:|---:|---:|---:|---:|']
    for tag,label in [('published','复现原折内排名'),('held_random','用其他随机订单校准阈值'),('held_ordered','用其他连续订单段校准阈值'),('historic_to_fresh','历史订单校准，历史以外订单检验')]:
        z=results[tag]['all']
        lines.append(f"| {label} | {z['selected_errors']} / {z['selected']} | {z['other_errors']} / {z['other']} | {fmt(z['rate_selected'])} | {fmt(z['rate_other'])} | {fmt(z['recall'])} |")
    lines+=['','在逐个留出随机或连续订单段时，每个询问订单和用于估计其阈值的订单都来自同一底层模型留出折，且整笔订单互不相交。八个连续段被选与未选错误率之差介于14.7至30.8个百分点；八个随机段介于20.3至30.3个百分点。七项底层分数在这些检验中沿用同一次整单拟合。','',
           '## 哪些方向能够单独验证','',
           '| 行为与父版预测 | 被选错误 / 人数 | 未选错误 / 人数 |','|---|---:|---:|']
    for key,label in [('awake_parent_0','清醒有消费，预测未传送'),('awake_parent_1','清醒有消费，预测传送'),('sleep_parent_0','休眠无已记录正消费，预测未传送'),('sleep_parent_1','休眠无已记录正消费，预测传送')]:
        z=results['published'][key]
        lines.append(f"| {label} | {z['selected_errors']} / {z['selected']} | {z['other_errors']} / {z['other']} |")
    az=results['published']['order_ci']['all']
    lines+=['',f"按行为、父版预测方向、单人或多人订单分层，用被选乘客的人群构成加权。可比较各层的被选错误率高出{100*az['adjusted_difference']:.1f}个百分点，按整单重抽估计的95%区间为{100*az['adjusted_95'][0]:.1f}至{100*az['adjusted_95'][1]:.1f}个百分点。休眠无正消费且预测未传送的31人全部被选，这个子群缺少同方向对照。历史订单以外，793名符合两类条件者中仅43人来自多人订单，检验主要反映独行乘客。",'',
           '## 分母核对','',
           '8,532名可核验父版的乘客中，7,020人的七项预测均支持该父版，合计862错。其中清醒有消费或休眠无正消费者共有6,518人、776错。要求这些人还同意早先111人版时，七票人群变为7,011人、860错。另有14人的七项预测内部相同、却反对固定父版，其中11人属于上述两种行为。','',
           '七项预测包含三种基础模型的中位数和较早模型的中位数，彼此相关。历史以外订单明确未进入原历史档案，但其汇总成绩已在项目评价中用过。置信区间按既定人群和四分之一预算计算；同一批数据上的其他预算及人群尝试会影响外推。']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('denominators',rr['denominators'])
    for tag,entry in results.items():
        print(tag,entry['all'],entry['historical_outside'],entry['order_ci']['all'],flush=True)


if __name__=='__main__':run()
