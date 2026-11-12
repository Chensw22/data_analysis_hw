# 第30轮：消费方向的固定单调先验

v60固定三折81.3920%，匹配无约束v59为81.5933%，原v34为81.9241%。先验没有收益，当前最佳仍v44 81.9816%，83%门槛未通过，无线上操作。

两实验模型都只移除TotalSpend/AllSpendingZero/PaidServices，并明确max_ctr_complexity=1；约束版额外设置Food/Shopping原值及log为正、Room/Spa/VR原值及log为负、LuxurySpend负、两对比正（共13列）。其他Cat与空间专家设置相同。不搜索符号或强度，不把相关性当因果。

7项测试和真实512训练/1152场景probe通过：重建全部衍生特征后28,305次原金额方向检查零违规，覆盖546个含缺失行及124个空间路由行，joblib逐值一致。官方CatBoost源码验证max_ctr_complexity=1排除cat+float CTR绕过路径。单调性只适用于观测金额变化且其它属性和缺失模式固定；不包含NaN→observed的保证。完整探索证据与限制见上级monotone_prior_research.json。
