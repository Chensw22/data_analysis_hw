"""Order-level retrospective transfer audit. All public inputs are scores, not passenger outcomes."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit,logit,ndtr
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from audit_flip_capacity_n23 import correct_count
from frozen_queue_shared_errors_n26 import RELEASES
from full_roster_booking_validation_n23 import ROOT,data
from repeated_archive_misses_n26 import ART,cohort

DEST=ART/'nov27_receipt_transfer_uncertainty'
STUDY=ROOT/'reports/nov23_matched_learning_curve'
REPORT=STUDY/'paired_public_transfer_uncertainty.json'
NAMES=['proposal_n26_champion_fare_six','proposal_n26_joint_fare_foundations','proposal_n27_unpaid_group_transfer',
       'proposal_n27_directional_foundation','proposal_n27_independent_tabnet','proposal_n27_joint_masked_fare',
       'proposal_n27_three_payment_states','proposal_n27_independent_payment_arbitration','proposal_n27_receipt_mitra_party_union',
       'proposal_n27_joint_error_experts']
NINE=RELEASES['nine']
DRAWS=16000


def read(file):return json.loads(file.read_text())


def official(name,ids):
    folder=ROOT/'reports'/name
    attempts=[read(p) for p in (folder/'online_attempts').glob('*.json')]
    entry=next(v for v in attempts if v.get('evaluation',{}).get('Status')=='COMPLETE')
    csv=ART/name/'submission.csv'
    assert hashlib.sha256(csv.read_bytes()).hexdigest()==entry['file_sha256']
    t=pd.read_csv(csv)
    np.testing.assert_array_equal(t.PassengerId,ids)
    assert t.Transported.dtype==bool
    return t.Transported.to_numpy(),correct_count(entry['evaluation']['Public Score'],len(ids)),entry['submission_ref']


def late_history():
    from frozen_release_order_mix_n26 import collect,sources,NAMES as old,FIRST,apply,fit,historical_training
    path=DEST/'late_history.npz'
    if not path.exists():
        runs,y,g,f,v=collect()
        q=np.concatenate([r[0] for r in runs['full']]);last=np.concatenate([p^(same&act['actual_and_scores']) for _,same,p,act in runs['full']])
        assert len(q)==len(set(q))==8532
        hist=sources()[-1]['historical'];ix=[list(old).index(k) for k in [*FIRST,'review','fare','six']]
        veto=apply(v[q][:,ix],fit(*historical_training(y,g,hist)))
        np.savez_compressed(path,q=q,ids=pd.read_csv(ROOT/'data/raw/train.csv').PassengerId.to_numpy(str)[q],labels=y[q],fold=f[q],six=v[q,-1],tabnet=last,reserve=last^veto)
    return path


def load():
    raw,y,g,f=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns);frame=pd.concat([raw,test],ignore_index=True)
    ids=frame.PassengerId.to_numpy(str);groups=frame.PassengerId.str.split('_').str[0].to_numpy()
    ref,_,known,fresh=cohort(raw,y,f)
    def get(path,key,*cols):
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z[key],ids)
            return tuple(z[c] for c in cols)
    names,bank=get('nov27_unsubmitted_audit/archived_releases.npz','ids','names','pred')
    pred={RELEASES[k]:bank[:,i] for i,k in enumerate(names)}
    run=read(STUDY/'official_submissions_2026-11-27.json')['results']
    assert [r['proposal'] for r in run]==NAMES
    first,med=get('nov27_dissent_likelihood_transfer/aligned.npz','passenger','first','random_median')
    fixed=first^med
    pred[NAMES[0]]=first
    ref_test=official(NAMES[1],ids[n:])[0]
    pred[NAMES[1]]=np.r_[ref,ref_test]
    root=pred[NINE]
    paid=frame[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].gt(0).any(axis=1).to_numpy()
    multi=frame.PassengerId.str[:4].duplicated(keep=False).to_numpy()
    archived=(pred[RELEASES['fields']]!=root)|(pred[RELEASES['band']]!=root)
    pred[NAMES[2]]=first^(multi&~paid&(first==root)&archived)
    pred[NAMES[3]]=fixed
    tabnet,=get('nov27_outsider_directional/aligned.npz','ids','TabNet_retained_random')
    pred[NAMES[4]]=fixed^tabnet
    missing,=get('nov27_masked_reference_receipts/replica/aligned.npz','passenger','action_replicate_independent_shift')
    up,=get('nov27_fifty_fare_adjudication/scores.npz','ids','level_fifty_random_up')
    pred[NAMES[5]]=fixed^(missing|up)
    route,anchor=get('nov27_payment_state_routes/aligned.npz','passenger','random','parent')
    np.testing.assert_array_equal(anchor,fixed)
    route=route.any(1);pred[NAMES[6]]=fixed^route
    opinions,=get('nov27_manifest_context_coding/aligned.npz','ids','prob')
    pred[NAMES[7]]=fixed^(route&((opinions[:,2]>=.5)!=fixed))
    m,=get('nov27_mitra_fare/aligned.npz','ids','111_specific_with_support')
    ra,oa=get('nov27_complete_receipt_counterfactual/aligned_random_2.npz','ids','average_random_calibrated','average_ordered_calibrated')
    party,=get('nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz','ids','random')
    receipt=(m&(first==fixed))|(ra&oa)|party
    pred[NAMES[8]]=fixed^receipt
    more,=get('nov27_unsubmitted_audit/retained_party_fields_41.npz','ids','random')
    pred[NAMES[9]]=fixed^(receipt|more)
    assert read(ROOT/'reports'/NAMES[9]/'selection.json')['selected']=='receipt_party'
    old=read(STUDY/'nov26_official_queue_results.json')['scores']
    with np.load(late_history()) as z:
        q=z['q'];np.testing.assert_array_equal(np.sort(q),np.flatnonzero(known))
        for k,expected in [('ids',ids[q]),('labels',y[q]),('fold',f[q]),('six',pred[RELEASES['six']][q])]:
            np.testing.assert_array_equal(z[k],expected)
        for k in ['tabnet','reserve']:
            name=old[k]['proposal']; tmp=root.copy();tmp[q]=z[k]
            tmp[n:]=official(name,ids[n:])[0];pred[name]=tmp
    days={}
    for day,edges in [('2026-11-26',[(v['proposal'],NINE) for v in old.values()]),('2026-11-27',[(v['proposal'],v['compared_parent']) for v in run])]:
        assert len(edges)==10
        contrasts=[]
        for ch,par in edges:
            cq,cc,ref=official(ch,ids[n:]);pq,pc,_=official(par,ids[n:])
            np.testing.assert_array_equal(cq,pred[ch][n:]);np.testing.assert_array_equal(pq,pred[par][n:])
            local=pred[ch][:n]!=pred[par][:n];diff=cq!=pq
            obs=cc-pc;mtest=int(diff.sum())
            assert abs(obs)<=mtest and (obs+mtest)%2==0 and mtest>0
            contrasts.append(dict(name=ch,parent=par,ref=ref,local=local&known,side=pred[ch][:n],test=diff,test_side=cq,
                       obs=obs,size=mtest,tpos=(obs+mtest)//2,unknown_edit_pattern_not_certified=int((~known).sum())))
        days[day]=contrasts
    return days,y,groups[:n],groups[n:],f,known,fresh


def count(data,mask,y,groups):
    a=data['local']&mask
    right=data['side']==y
    return dict(edits=int(a.sum()),orders=int(len(np.unique(groups[a]))),fixed=int((a&right).sum()),harmed=int((a&~right).sum()))


def posterior(edges,y,groups,mask,rng,draws):
    # Shared Exp(1) weights for whole bookings preserve dependence between estimates. Independent
    # Gamma(.5) pseudo-counts in each directional cell give finite uncertainty at all-success/all-failure boundaries.
    active=np.any([v['local']&mask for v in edges],axis=0)
    booking,rev=np.unique(groups[active],return_inverse=True)
    mat=np.zeros((len(booking),len(edges),2,2),np.float32)
    target=[]
    for j,v in enumerate(edges):
        ix=np.flatnonzero((v['local']&mask)[active])
        side=v['side'][active][ix].astype(int)
        right=(v['side'][active][ix]==y[active][ix]).astype(int)
        np.add.at(mat,(rev[ix],j,side,right),1)
        target.append(np.bincount(v['test_side'][v['test']].astype(int),minlength=2))
    totals=mat.sum(0)
    probs=np.empty((draws,len(edges),2))
    for lo in range(0,draws,256):
        hi=min(draws,lo+256)
        w=rng.exponential(size=(hi-lo,len(booking))).astype(np.float32)
        cnt=(w@mat.reshape(len(booking),-1)).reshape(hi-lo,len(edges),2,2)
        cnt=cnt+rng.gamma(.5,size=cnt.shape)
        probs[lo:hi]=cnt[:,:,:,1]/cnt.sum(axis=3)
    target=np.array(target)
    mean=(totals[:,:,1]+.5)/(totals.sum(axis=2)+1)
    return probs,mean,target,totals


def limits(v):
    return [float(x) for x in np.quantile(v,[.025,.975])]


def roll(p,orders,side,rng):
    # Common uniform for all changes within a booking gives the perfect positive within-order success-dependence limit.
    _,ix=np.unique(orders,return_inverse=True)
    count=np.zeros((ix.max()+1,2),int);np.add.at(count,(ix,side.astype(int)),1)
    groups,num=np.unique(count,axis=0,return_counts=True)
    out=np.zeros(len(p),int)
    for (a,b),m in zip(groups,num):
        if a==0 or b==0:
            out+=int(a+b)*rng.binomial(int(m),p[:,int(a==0)])
        else:
            high=p.max(1);low=p.min(1)
            both=rng.binomial(int(m),low)
            one=rng.binomial(int(m)-both,np.divide(high-low,1-low,out=np.zeros(len(p)),where=low<1))
            out+=int(a+b)*both+np.where(p[:,0]>p[:,1],a,b)*one
    return 2*out-len(side)


def holm(ps):
    p=np.asarray(ps);order=np.argsort(p)
    output=np.empty_like(p);output[order]=np.maximum.accumulate(np.minimum(1,(len(p)-np.arange(len(p)))*p[order]))
    return output


def analyze(edges,y,groups,tgroups,mask,fresh,seed,draws=DRAWS):
    rng=np.random.default_rng(seed)
    pp,mean,counts,local=posterior(edges,y,groups,mask,rng,draws)
    param=((2*pp-1)*counts).sum(2)
    pred=((2*mean-1)*counts).sum(1)
    ps=[];records=[]
    for j,v in enumerate(edges):
        m=v['test'];o=v['obs'];p=pp[:,j,:];target=counts[j]
        sampled=roll(p,tgroups[m],v['test_side'][m],rng)
        iid=2*(rng.binomial(target[0],p[:,0])+rng.binomial(target[1],p[:,1]))-v['size']
        tail=min(1.,2*min(float(np.mean(sampled<=o)),float(np.mean(sampled>=o))))
        ps.append(tail)
        records.append(dict(name=v['name'],parent=v['parent'],ref=v['ref'],local=count(v,mask,y,groups),new_orders=count(v,mask&fresh,y,groups),
            target_changes=v['size'],target_orders=int(len(np.unique(tgroups[m]))),target_fixed=int(v['tpos']),official_net=o,
            expected_net=float(pred[j]),precision_by_new_class=mean[j].tolist(),local_changed_by_new_class=local[j].sum(1).astype(int).tolist(),
            target_by_new_class=target.tolist(),local_expected_95=limits(param[:,j]),predictive_95_whole_booking=limits(sampled),
            predictive_95_independent_passengers=limits(iid),predictive_tail_two_sided=tail,
            inside_95=bool(np.quantile(sampled,.025)<=o<=np.quantile(sampled,.975))))
    for row,h in zip(records,holm(ps)):row['holm_tail_within_day']=float(h)
    return records,param


def portfolio(edges,records,param,tgroups,indices):
    # The sum of edge gains is one portfolio outcome. Opposite directions on the SAME passenger cancel.
    # The remaining unknown outcome contributes ±weight. For arbitrary within-order dependence the
    # maximum standard deviation assuming independent bookings is sqrt(sum_g (sum_j |weight_j|)^2).
    sig=np.stack([np.where(e['test'],np.where(e['test_side'],1,-1),0) for e in edges])[indices]
    exposure=np.abs(sig.sum(0))
    _,ix=np.unique(tgroups,return_inverse=True);by_order=np.bincount(ix,weights=exposure)
    l2=float(by_order@by_order)
    obs=sum(records[i]['official_net'] for i in indices)
    expected=sum(records[i]['expected_net'] for i in indices)
    sampled=param[:,indices].sum(1)
    low=float(np.quantile(sampled,.025));gap=max(0.,low-obs)
    # A many-independent-booking normal approximation using the largest possible target variance.
    # Enlarge the variance further by the actual joint OOF resampling error. This is separate from
    # the bounded-variable inequality and does not synthesize individual test labels.
    worst_sd=np.sqrt(l2)
    approx_sd=np.sqrt(l2+sampled.var(ddof=1))
    normal_tail=float(ndtr((obs-expected)/approx_sd))
    mixture_tail=float(np.exp(-np.maximum(0.,sampled-obs)**2/(2*l2)).mean()) if l2 else 1.
    return dict(conditional_hoeffding_tail_averaged_over_booking_bootstrap=mixture_tail,
        largest_order_weight=int(by_order.max()),booking_bootstrap_expected_sd=float(sampled.std(ddof=1)),
        approximate_joint_sd_upper=float(approx_sd),normal_approximation_one_sided_tail=normal_tail,
        conservative_variance_normal_95=[float(expected-1.96*approx_sd),float(expected+1.96*approx_sd)],
        max_single_order_weight_over_target_sd_upper=float(by_order.max()/worst_sd) if worst_sd else 0.,
        comparisons=len(indices),total_comparison_changes=int(abs(sig).sum()),unique_changed_passengers=int((np.abs(sig).sum(0)>0).sum()),
        unique_orders=int(len(np.unique(tgroups[abs(sig).sum(0)>0]))),same_person_max_multiplicity=int(np.abs(sig).sum(0).max()),
        signed_weight_after_cancel=int(exposure.sum()),realized_sum_of_nets=int(obs),expected_sum_of_nets=float(expected),
        expected_sum_95=limits(sampled),target_sampling_sd_upper=float(np.sqrt(l2)),observed_minus_expected=float(obs-expected),
        one_sided_hoeffding_plus_estimation_tail=min(1.,.025+float(np.exp(-gap**2/(2*l2)))) if l2 else 1.,
        explanatory_note='The last quantity combines an approximate one-sided 97.5% booking-bootstrap lower bound for the expectation with a conservative finite-target Hoeffding tail. It is a sensitivity bound under transferable expected rates and independent bookings, not a calibrated formal test.')


def run():
    DEST.mkdir(exist_ok=True)
    days,y,g,tg,fold,known,fresh=load()
    report={};parts={}
    for i,(day,edges) in enumerate(days.items()):
        records,param=analyze(edges,y,g,tg,known,fresh,57181+i)
        portfolio_all=portfolio(edges,records,param,tg,list(range(len(edges))))
        point=np.array([r['expected_net'] for r in records]);truth=np.array([r['official_net'] for r in records]);size=np.array([r['target_changes'] for r in records])
        val=dict(edges=records,portfolio=portfolio_all,marginal_coverage=sum(r['inside_95'] for r in records),
               number_expected_over_actual=int((point>truth).sum()),raw_mae_nets=float(abs(point-truth).mean()),no_gain_mae_nets=float(abs(truth).mean()))
        if i:
            historical=parts['2026-11-26']
            rates=((point/size)+1)/2
            corrected=(2*expit(logit(rates)+historical['offset'])-1)*size
            val['prior_day_offset_applied']=dict(log_odds_offset=historical['offset'],mae_nets=float(abs(corrected-truth).mean()),expected_nets=corrected.tolist())
            val['after_first_fare_parent']=portfolio(edges,records,param,tg,list(range(1,len(edges))))
            val['children_of_directional_parent']=portfolio(edges,records,param,tg,list(range(4,len(edges))))
        else:
            rates=((point/size)+1)/2
            desired=np.mean((truth/size+1)/2)
            off=brentq(lambda x:expit(logit(rates)+x).mean()-desired,-10,10)
            parts[day]=dict(offset=float(off))
            val['fitted_single_offset_for_later_day']=float(off)
        report[day]=val
    report['methods']={'training_rows':len(y),'verifiable_training_rows':int(known.sum()),'historically_unseen_verifiable_rows':int((known&fresh).sum()),
        'predictive_simulations':DRAWS,'seed':57181,'scores':'Verified all daily CSV hashes and publication receipts. The compatible denominator of 4277 allows aggregate signed changes to be recovered. Only official score aggregates are used.',
        'prediction':'Condition on how many incumbent disagreements end at each new class. Estimate success among changed OOF passengers in each direction, using a half success and half failure smoothing. Resample whole OOF orders jointly across models. Sample prospective outcomes conditionally on these rates with a common success rank per test booking; give independent-passenger sensitivity as well.',
        'limitations':['Forecasts are retrospective, after many local policies were screened. The random parent replay is the main OOF protocol. Edited orders, raw training passengers, and all public outcomes are reused for many releases. A comparison against a different score-tied parent would have different edits despite the identical public gain.',
        'The marginal predictive tails assume comparable directional edit success on held and public orders. Perfect positive within-order success dependence is a sensitivity case, not a proven distribution. Holm values use approximate marginal predictive tails.',
        'The 26th compares all ten releases to a shared nine-vote anchor, using earlier official summaries. It checks calibration on an earlier decision sequence but reuses the same public passengers and several proposal families, so it is not a separate held-out leaderboard.',
        'The later releases were partly chosen in light of earlier public aggregate scores. That adaptive dependence also limits interpreting a nominal tail probability as a prospective test. The 161 training rows without authenticated parent decisions are excluded. Historically unseen rows are part of the full OOF sample. Different random and contiguous calibrations on the same OOF scores are not treated as new experiments.',
        'Counts do not measure what proportion of OOF optimism is caused by model search, training-size effects, feature shift, or chance. Public comparisons cannot identify any passenger outcome.']}
    output_md=STUDY/'paired_public_transfer_uncertainty.md'
    latest=report['2026-11-27'];previous=report['2026-11-26'];pool=latest['portfolio']
    display=['111人票价与六票','双票价追加','多人无正消费','分方向基础模型','独立 TabNet','缺失账单与票价联合',
             '三种消费状态','消费分歧裁决','删项、Mitra 与同行专家','收据与多人规则']
    lines=['# 11月27日十份提交的离线到公榜迁移', '',
           '当天十项的公榜净收益都低于按离线改判率预测的中心值。整单预测区间覆盖其中八项。单次小规模版本即使出现负分，也可能与训练订单上的效果相容。', '',
           '| 版本 | 测试改判 | 整单留出改对 / 改错 | 预期公榜净增人数 | 正式净增 | 95%预测区间 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,row in zip(display,latest['edges']):
        lo,hi=row['predictive_95_whole_booking']
        lines.append(f"| {name} | {row['target_changes']} | {row['local']['fixed']} / {row['local']['harmed']} | {row['expected_net']:+.1f} | {row['official_net']:+d} | [{lo:.0f}, {hi:.0f}] |")
    lines+=['','按提交时记录的最高分父版比较。95%预测区间同时包含训练端估计误差和有限测试订单的抽样波动。仅使用官方汇总分数及已核验的4,277人分母计算线上净增。', '',
      '## 重复订单与整组偏移','',
      f"当天共有 {pool['total_comparison_changes']} 次父子改判，涉及 {pool['unique_changed_passengers']} 名乘客、{pool['unique_orders']} 笔订单。同一乘客最多被计算 {pool['same_person_max_multiplicity']} 次。保留相同乘客不同方向的抵消后，十项比较净增之和的离线预期为 {pool['expected_sum_of_nets']:+.1f}，公榜为 {pool['realized_sum_of_nets']:+d}。这个量描述所选十项比较，不能换算成一个新模型的准确率。", '',
      f"使用同一组整单权重重抽十项离线预测，并给测试订单采用最大可能方差，正态近似下整组单侧尾概率为 {pool['normal_approximation_one_sided_tail']:.3f}。仅假定不同测试订单独立、允许同一订单任意相关时，结合离线估计下界的宽松尾概率上界为 {pool['one_sided_hoeffding_plus_estimation_tail']:.3f}。两种计算都以离线条件改判率可以迁移作为参照，近似结果依赖分布假设。", '',
      f"当天两个落在预测区间外的版本分别是 TabNet 和缺失账单联合。对十个近似双侧尾概率做 Holm 校正后，最小值为 {min(v['holm_tail_within_day'] for v in latest['edges']):.3f}。", '',
      '## 前一日回测','',
      f"用同一套方法重放11月26日十个文件，均对照同一个九票基线。十项的离线预期也全部偏高，95%预测区间覆盖 {previous['marginal_coverage']} 项。其 {previous['portfolio']['total_comparison_changes']} 次改判只涉及 {previous['portfolio']['unique_changed_passengers']} 名乘客。两日共用公榜样本及部分相同模型家族。", '',
      f"直接用离线条件改判率预测，11月27日每项净得失的平均绝对误差为 {latest['raw_mae_nets']:.1f} 人。仅用26日的结果估计一个统一的对数几率偏移后，为 {latest['prior_day_offset_applied']['mae_nets']:.1f} 人，统一预测零增益为 {latest['no_gain_mae_nets']:.1f} 人。", '',
      '## 可检验范围','',
      '每次抽样以整笔订单为单位，在本地各版本共享权重。每个版本按最终改判为真或假分别估计成功率，对公榜用实际改判数量和每笔订单的改判人数。表内采用同一订单内完全同向波动的情形，逐乘客独立的敏感性结果另存 JSON。检验只使用8,532名能准确回放父版的训练乘客。', '',
      '历史未见的1,033人属于上述样本。随机和连续订单校准重叠，同一公榜上的十份提交也重叠。后续选模参考过之前的公榜汇总分数，尾概率应按回顾性诊断解读。这份审查无法依靠两天数据独立估计模型筛选、训练规模和人群变化各自造成多少偏差。']
    output_md.write_text('\n'.join(lines)+'\n')
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print([(k,v['marginal_coverage'],v['number_expected_over_actual'],v['portfolio']) for k,v in report.items() if 'edges' in v],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
