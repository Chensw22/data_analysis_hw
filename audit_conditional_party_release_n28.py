"""Paired whole-booking audit of the frozen conditional-allocation release against three score-tied releases."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from audit_flip_capacity_n23 import correct_count
from export_conditional_party_allocation_n27 import apply
from full_roster_booking_validation_n23 import ROOT,data
from paired_public_transfer_uncertainty_n27 import posterior,roll
from repeated_archive_misses_n26 import ART,cohort

TARGET='proposal_n27_conditional_party_allocation'
MODELS={'16':'proposal_n27_directional_foundation','68':'proposal_n27_three_payment_states',
        '18':'proposal_n27_receipt_mitra_party_union','59':'proposal_n27_joint_error_experts'}
REPORT=ROOT/'reports/nov23_matched_learning_curve/conditional_party_release_audit_n28.json'
REPS=12000


def official(name,test_ids):
    scope=ROOT/'reports'/name;manifest=json.loads((scope/'manifest.json').read_text())
    csv=ART/name/'submission.csv';digest=hashlib.sha256(csv.read_bytes()).hexdigest();assert digest==manifest['sha256']
    row=pd.read_csv(csv);np.testing.assert_array_equal(row.PassengerId,test_ids)
    attempts=[json.loads(path.read_text()) for path in (scope/'online_attempts').glob('*.json')]
    attempt,=[r for r in attempts if r['evaluation']['Status']=='COMPLETE' and r['file_sha256']==digest]
    return row.Transported.to_numpy(),dict(ref=attempt['submission_ref'],score=attempt['evaluation']['Public Score'],sha256=digest)


def counts(local,mask,candidate,truth):
    take=local&mask
    return dict(rows=int(mask.sum()),edits=int(take.sum()),fixed=int(np.sum(take&(candidate==truth))),harmed=int(np.sum(take&(candidate!=truth))))


def composition(edit,groups,size,new):
    ix=np.flatnonzero(edit)
    active,num=np.unique(groups[ix],return_counts=True)
    return dict(passengers=len(ix),orders=len(active),population_size={str(k):int(np.sum(size[edit]==k)) for k in sorted(np.unique(size[edit]))},
          edits_per_order={str(k):int((num==k).sum()) for k in sorted(np.unique(num))},
          to_true=int(new[ix].sum()),to_false=int((~new[ix]).sum()))


def run():
    raw,y,g,f=data();n=len(y)
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=raw.columns);frame=pd.concat([raw,test],ignore_index=True)
    ids=frame.PassengerId.to_numpy(str);groups=frame.PassengerId.str[:4].to_numpy()
    size=frame.groupby(groups).PassengerId.transform('size').to_numpy()
    _,blind,known,fresh=cohort(raw,y,f)
    def get(path,*keys):
        with np.load(ART/path) as z:
            np.testing.assert_array_equal(z['ids' if 'ids' in z else 'passenger'],ids)
            if 'y' in z:np.testing.assert_array_equal(z['y'],y)
            return tuple(z[k] for k in keys)
    test_decisions={};records={}
    for key,name in [('56',TARGET),*MODELS.items()]:
        test_decisions[key],records[key]=official(name,ids[n:])
    a16=test_decisions['16']
    with np.load(ART/TARGET/'inputs.npz') as z:
        np.testing.assert_array_equal(z['passenger'],ids[n:]);np.testing.assert_array_equal(z['parent']>=.5,a16)
        testscore=apply(z['parent'],z['prior_odds'],z['scores'],z['passenger'])
    np.testing.assert_array_equal(testscore>=.5,test_decisions['56'])
    csv=pd.read_csv(ART/TARGET/'test_probabilities.csv',float_precision='round_trip')
    np.testing.assert_array_equal(csv.probability,csv.reload_probability);np.testing.assert_array_equal(csv.probability,testscore)
    first,median=get('nov27_dissent_likelihood_transfer/aligned.npz','first','random_median')
    fixed=first^median
    opinion,=get('nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz','opinions')
    seven=((opinion>=.5)==fixed[:,None]).all(1)
    mitra,=get('nov27_mitra_fare/aligned.npz','111_specific_with_support')
    prior=np.where(fixed,1.,-1.)*np.maximum(1e-6,abs(logit(np.median(opinion[:,1:4],axis=1).clip(.002,.998))))
    packet=ART/TARGET/'inputs.npz'
    np.testing.assert_allclose(prior[n:],np.load(packet)['prior_odds'])
    unknown_group=np.isin(g,g[~known]);complete_known=~unknown_group
    all_orders=np.unique(g);ordered=np.zeros(n,int)
    for k,order in enumerate(np.array_split(all_orders,8)):ordered[np.isin(g,order)]=k
    target=test_decisions['56'];ref_count=correct_count(records['56']['score'],len(test))
    overlaps={}
    for tag,v in test_decisions.items():
        if tag=='56':continue
        own=target!=a16;other=v!=a16;diff=v!=target
        score_delta=ref_count-correct_count(records[tag]['score'],len(test))
        assert (diff.sum()+score_delta)%2==0
        overlaps[tag]=dict(distance=int(diff.sum()),changed_vs_16=int(other.sum()),overlap_passengers=int((own&other).sum()),
            overlapping_orders=int(len(np.intersect1d(groups[n:][own],groups[n:][other]))),
            candidate_vs_other=composition(diff,groups[n:],size[n:],target),official_net=score_delta,
            public_fixed=int((diff.sum()+score_delta)//2),public_harmed=int((diff.sum()-score_delta)//2))
    report={'official':records,'test_comparisons':overlaps,'known_rows':int(known.sum()),'unknown_rows':int((~known).sum()),
        'fresh_distribution':dict(all=int((fresh&known).sum()),multi=int((fresh&known&(size[:n]>1)).sum()),
            three_or_more=int((fresh&known&(size[:n]>=3)).sum()))}
    for i,(scheme,fold) in enumerate([('random',f),('ordered',ordered)]):
        score,stored=get('nov27_conditional_party_allocation/aligned.npz',scheme+'_scores',scheme+'_supported_increment')
        replay=apply(fixed.astype(float),prior,score,ids)>=.5
        np.testing.assert_array_equal(replay,stored^fixed);np.testing.assert_array_equal(replay[n:],target)
        route,route_par=get('nov27_payment_state_routes/aligned.npz',scheme,'parent')
        np.testing.assert_array_equal(route_par,fixed)
        cf1,cf2=get(f'nov27_complete_receipt_counterfactual/aligned_{scheme}_2.npz','average_random_calibrated','average_ordered_calibrated')
        peers,=get('nov27_broad_population_expert_crosscheck/unanimous_party_actions.npz',scheme)
        receipt=(mitra&(first==fixed))|(cf1&cf2)|peers
        parents={'16':fixed,'18':fixed^receipt,'68':fixed^route.any(1)}
        result={};all_delta=[];episodes=[]
        condknown=(size[:n]>1)&complete_known
        err16=(fixed[:n]!=y)
        bad_order=np.isin(g,g[complete_known&err16])
        for tag,anchor in parents.items():
            np.testing.assert_array_equal(anchor[n:],test_decisions[tag])
            change=anchor[:n]!=replay[:n]
            diff=change*np.where(replay[:n]==y,1,-1)
            np.testing.assert_array_equal(diff,(replay[:n]==y).astype(int)-(anchor[:n]==y).astype(int))
            scopes={'all':known,'fresh':fresh&known,'multi':known&(size[:n]>1),'seven':known&seven[:n],'prior_31':blind,
                    'complete_known_orders':complete_known,
                    'complete_multi':condknown,'posthoc_order_with_parent16_error':condknown&bad_order,
                    'posthoc_order_parent16_all_correct':condknown&~bad_order}
            out={s:counts(change,m,replay[:n],y) for s,m in scopes.items()}
            unknown_changes=change&~known
            out['composition']=composition(change&known,g,size[:n],replay[:n])
            out['conditional_on_cached_decisions_unknown_rows']=int(unknown_changes.sum())
            out['fold_net']=[int(diff[known&(fold==j)].sum()) for j in range(8)]
            # One uncertified companion can change another member's inferred allocation.
            # Only the wholly known orders supply a fixed contribution for this worst-case bound.
            uncertain=int(unknown_group.sum())
            complete_net=int(diff[complete_known].sum())
            out['unrestricted_unknown_order_bounds']=[complete_net-uncertain,complete_net+uncertain]
            out['known_members_in_partly_unknown_orders']=counts(change,known&unknown_group,replay[:n],y)
            out['by_new_class']={str(k):counts(change,known&(replay[:n]==k),replay[:n],y) for k in [False,True]}
            out['all_fold_net']=int(diff[known].sum())
            active=change&known;aa=np.flatnonzero(active);orders,idx=np.unique(g[aa],return_inverse=True)
            effect=np.bincount(idx,weights=diff[aa]);out['orders_helped_tied_harmed']=[int((effect>0).sum()),int((effect==0).sum()),int((effect<0).sum())]
            all_delta.append(diff*known);result[tag]=out
            episodes.append(dict(local=active,side=replay[:n],test=anchor[n:]!=target,test_side=target))
        # Draw common weights on full bookings so all parent contrasts retain their dependence.
        useful=np.any(np.array(all_delta)!=0,axis=0);ordids,rev=np.unique(g[useful],return_inverse=True)
        table=np.stack([np.bincount(rev,weights=v[useful]) for v in all_delta],axis=1)
        rng=np.random.default_rng(18144+i);nb=len(np.unique(g[known]))
        weights=rng.multinomial(nb,np.r_[np.full(len(ordids),1/nb),1-len(ordids)/nb],size=REPS)[:,:-1]
        resampled=weights@table
        for j,tag in enumerate(parents):result[tag]['local_net_booking_bootstrap_95']=np.quantile(resampled[:,j],[.025,.975]).tolist()
        pp,mean,target_cells,lcounts=posterior(episodes,y,g,known,rng,REPS)
        for j,(tag,e) in enumerate(zip(parents,episodes)):
            take=e['test'];sim=roll(pp[:,j,:],groups[n:][take],target[take],rng)
            result[tag]['public_projection']=dict(expected_net=float(((2*mean[j]-1)*target_cells[j]).sum()),predictive_95=np.quantile(sim,[.025,.975]).tolist(),
                directions=['to_false','to_true'],local_support=lcounts[j].sum(1).astype(int).tolist(),test_changes=target_cells[j].tolist())
        report[scheme]=dict(against=result,all_three_positive_in_shared_booking_bootstraps=float(np.mean((resampled>0).all(1))),
               order_cutoffs=['Each of the eight folds excludes whole bookings in this allocation model. Both protocols share the original parent and foundation prior.'])
    cohort_stats=report['random']['against']['16']
    source=json.loads((ROOT/'reports/nov23_matched_learning_curve/conditional_party_allocation.json').read_text())
    assert [cohort_stats['all']['fixed'],cohort_stats['all']['harmed']]==[source['random']['rules']['supported_increment']['scopes']['all'][k] for k in ['fixed','harmed']]
    report['method_notes']=['Compared actual 56-child and actual other release decisions directly. The candidate is always attached to the registered 16-change parent; no parent replacement or re-optimization.',
        'Training uses the observed total transported in labeled training orders for a conditional likelihood. Query allocation sums over every possible total given the frozen probabilities. Decision reconstruction for both OOF protocols and 4,277 competition passengers reads no query outcomes. Full-coverage denominators compare wrong-to-right and right-to-wrong.',
        'Orders containing parent16 errors are a diagnostic outcome-defined slice only. Its complement is included in the primary paired comparison. Restricting to this slice or only to wrong rows gives an optimistic impression. The rule and four candidate variants were examined using the same validation set; the paired intervals condition on that choice.',
        'Bootstrap resamples whole known bookings jointly across contrasts. The separate prospective-style interval conditions on test changes to each output class and uses common within-order outcome ranks, as in paired_public_transfer_uncertainty_n27.py. The official result is already known; these are descriptive retrospective checks.',
        'The official net against each score-tied parent is the same but the number of changed rows, and hence aggregate fixes and harms, differ. No test labels or individual test outcomes are reconstructed. Unknown OOF baseline rows are excluded; unrestricted numerical bounds are conservative.']
    placebo=json.loads((ROOT/'reports/nov23_matched_learning_curve/conditional_party_placebo.json').read_text())
    report['selection_sensitivity']={}
    for scheme in ['random','ordered']:
        def gains(entries):
            return {k:int(r['scopes']['all']['fixed']-r['scopes']['all']['harmed']) for k,r in entries.items()}
        report['selection_sensitivity'][scheme]=dict(original_four_rules=gains(source[scheme]['rules']),earlier_placebos=gains(placebo[scheme]))
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 56人版对三份82.370%父版的独立核验','',
           '正式回执59056549为82.183%。56人版在同一公榜上比三份82.370%文件都少判对8人。冻结模型对4,277名测试乘客的概率逐项回放一致。','',
           '| 比较父版 | 随机订单修 / 伤 | 随机净增95%区间 | 连续订单修 / 伤 | 连续净增95%区间 | 测试分歧 | 公榜修 / 伤 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for tag in ['16','68','18']:
        a,b=[report[k]['against'][tag] for k in ['random','ordered']];pub=overlaps[tag]
        def ratio(d):return f"{d['all']['fixed']} / {d['all']['harmed']}"
        def ci(d):return '['+', '.join(str(int(v)) for v in d['local_net_booking_bootstrap_95'])+']'
        lines.append(f"| {tag}人版 | {ratio(a)} | {ci(a)} | {ratio(b)} | {ci(b)} | {pub['distance']} | {pub['public_fixed']} / {pub['public_harmed']} |")
    lines+=['','局部收益按8,532名可核验乘客计算，置信区间以订单为单位联合重抽12,000次。六个区间都包含零。两种切法均重新训练了分配模型，但共用既有的父版和基础模型分数。以真实测试改判方向从离线预测公榜，相对16人版的期望约为净增6至7人，两种切法的预测区间均为−14至+26人。','',
           '## 实际改动及覆盖','',
           '相对16人版的56项改判全部来自多人订单，共48笔，两个方向各28人。41笔改动一人、6笔改动两人、1笔改动三人。与59、68、18人版的原始改判分别重叠13、7、3人，最终文件的标签分歧分别为89、110、68人。同一父版16的改判方向在训练和测试并不同，随机验证为51人改为传送、23人改为未传送，测试为28/28。','',
           '历史未见的1,033名可核验乘客中，55人来自双人订单，无人来自三人及以上订单。56人版在这1,033人中完整保留16人版的结果，未能检验其对新多人订单的泛化。8,693人中161人的原父版无法认证，且他们的同行也会影响组内分配。限制到全单成员均可核验的8,343人时，56人版对16人版随机37修20伤、连续31修19伤。','',
           '## 是否选择了父版已经出错的乘客','',
           '模型在有标签的训练订单上学习同行之间谁更可能传送。对留出和正式测试订单，程序从冻结的个人先验计算各个订单传送人数的概率，并对所有可能人数求和。分配和双模型同意条件均可在不读取查询标签时回放。','',
           '如果事后只数父版16已犯错的完整多人订单，随机验证为37修11伤，父版原本全对的订单还贡献0修9伤。连续验证相应为31修8伤和0修11伤。主表保留这两部分。七路都赞成父版时仅1修4伤、3修3伤。四个离线规则在同一批验证样本上比较后选中了此规则，随机与连续是对同一批人的重算。连续验证中三种替换同行的对照净增4、9、16人，真实同行净增11人，无法仅归因于真实同行。','',
           '公榜改对、改错人数由整体分差和文件分歧数计算，使用已核验的4,277人计分分母。它们不能区分具体是哪一名测试乘客判对。']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    for tag in ['16','68','18']:
        print(tag,report['test_comparisons'][tag],[(s,report[s]['against'][tag]['all'],report[s]['against'][tag]['local_net_booking_bootstrap_95'],report[s]['against'][tag]['public_projection']) for s in ('random','ordered')],flush=True)


if __name__=='__main__':
    with threadpool_limits(1):run()
