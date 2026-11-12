# 调研与审查交叉评议

参与者：research_improvements、review_current；主 agent 独立核验关键数据结论。

## 已核实

- 官方 train/test 的 GroupId 前缀和完整 Cabin 交集均为零。真实测试是新群组，不存在“训练集中有该测试乘客同行标签”的条件。
- 当前 v6 模型使用 Plain boosting、one_hot_max_size=2。六个类别字段基数都大于2，因此其内部均走CTR编码。
- 当前无ID基线未发现直接目标泄漏；已有随机CV81.58%、群组CV81.61%，原结果仍有效。
- 重复CV只是稳定性证据，不能估算进前10%的成功概率。

## 分歧与反证

主agent提出过严格内层OOF的同行/舱位标签统计，调研agent发现train/test群组无重叠，审查agent独立复验，双方否决此路线。即使编码无直接泄漏，也会与实际测试可用信息不一致。

v11同时修改特征和参数，不能单凭退步断言所有群组特征有害；v13/v14的同参数HGB对照下降约0.68个百分点，是较清楚的反证。以后尽量采用单因素对照。

## 下一轮

1. 固定v6所有其他设置，仅改one_hot_max_size=10。
2. 固定v6所有其他设置，仅改boosting_type=Ordered。
3. 依据单项结果决定组合或去掉重复的消费log特征。
4. 确认/导出时核验源码、配置哈希，防止旧gate用于新模型。
5. 超过83%的候选再进行两套新随机划分和群组隔离确认；不重新打开旧留出集，不改容易的划分来过门槛。

依据：[CatBoost类别特征说明](https://catboost.ai/docs/en/features/categorical-features)、[CatBoost参数文档](https://catboost.ai/docs/en/references/training-parameters/common)。没有证据保证下一轮超过83%。
