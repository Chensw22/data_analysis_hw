# 83% 门槛与历史预测复核

在已查到的历史文件中，没有可重放、按原三折覆盖全部 6,954 名乘客、准确率严格超过 83% 的待提交方案。门槛需要至少 5,772 人预测正确。已核对的最高可重放结果是 v116 的 5,711 人，差 61 人。该版本已提交。

## 可以复用的逐行预测

| 版本 | 本地正确人数与准确率 | 复用的文件与键 | 已测公开分 |
|---|---:|---|---:|
| v116 五家多数投票与纠错专家 | 5,711，82.1254% | artifacts/composed_local_error_routes.npz，five_family_majority_with_specialist，布尔决定 | 80.944%，58539379 |
| v93 Mitra 3,072 个支持样本 | 5,710，82.1110% | artifacts/run_056/v93_Mitra_support3072_seed20261112.npz，prob，软概率 | 81.295%，58454737 |
| v108 同行消费校正 | 5,663，81.4351% | artifacts/v108_exact_oof/oof.npz，v108，软概率 | 81.646%，58503435 |
| v131 当前公开最佳 | 5,662，81.4208% | artifacts/nov21_case_checks/v136_final_oof.csv，v131，布尔决定 | 81.692%，58653601 |

以 v131 CSV 的 row、fold、true 为参考，v116 的 dev/fold、v93 与 v108 的 dev_ids/fold_id 逐行一致。v93 的三份 fold*_rows.npz 中，实际训练和评估乘客恰好各自等于相应折外和折内行号，二者无交集。v116 的三个原折分别正确 1,915、1,902、1,894 人，仓库 export_majority_specialist.py 验证开发集规则的重放。这个组合复用了同一开发集做方案选择，其分数仍有选择偏差。v44 的 81.9816% 可通过 error_overlap_diagnostic.load_authenticated() 取出 v44 概率，早期单独数组没有自带折号。

## 容易误读的超过 83% 的数字

| 原始记录 | 需检查的分母、标签或信息来源 |
|---|---|
| reports/nov21_rescue_coverage/result.json、reports/error_rescue_20261117/catalog.json | 83.4196% 与 92.1484% 都在看见每个人的真实标签后挑出正确模型。这衡量错误覆盖范围，无法按此规则作盲测预测。 |
| reports/attribute_mask_research.json、reports/cryo_marginal_probe | 96.13% 衡量找回被遮挡的 CryoSleep 属性，目标变量与 Transported 不同。 |
| reports/data_error_diagnosis_20261117/error_slices.csv、reports/error_rescue_20261117/structure_slices.csv | 特定人群的条件准确率，分母是一部分乘客。 |
| reports/error_case_retrieval/operating_points.csv、reports/error_evidence_challengers | raw_original_correct 等选择条件本身依赖结果标签。诊断分层可以使用，部署时无法事先辨认。 |
| reports/error_targeted_20261117/RESULTS.md、reports/run_002/holdout_metrics.json | 单折只有 2,318 人，历史留出集是另外 1,739 人；均无法证明覆盖原三折。早期留出报告为 81.196%，此次没有读取其原始标签。 |

审计检索了历史报告中的 JSON/CSV 准确率和上界字段，核对 Markdown 高分上下文，并扫描当时 589 个 NPZ 的数组头。68 个文件自带与原验证集一致的行号和折号，其中筛出 386 个完整的一维布尔或浮点值向量（含标签及诊断掩码）。在这些向量中仅四份 y 真值本身在与标签比较时超过 83%。未显式携带验证身份的旧缓存，即使算出高分也不能凭长度当作合格模型。以上结论限于已有输出与可识别的评估字段，未把未运行代码、任意新组合或多维内部特征自动当作已验证的模型。

## 是否存在突破 85% 的样本覆盖

以 v131、v116 和 v93 的实际折外判断作三者多数票，正确 5,704 人（82.0247%）。逐人事后挑对的上界为 5,941 人（85.4328%）。加入本轮独立供体树叶近邻，四者上界达到 6,139 人（88.2801%），仍有 815 人四者均错，其中 489 人属于原 546 名共同错例。相对于前三版，近邻独家覆盖了 198 人。应在独立划分下检验是否能在预测前识别这些乘客，同时计算误伤；仅凭上界达标不能提交。
