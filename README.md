# Spaceship Titanic 课程项目

目标：三人团队稳妥完成可复现的竞赛项目，保留基线、改进、失败尝试和答辩证据。

2026-11-18共同错例专项已完成：错误识别器、同行联合边际推断均无整体增益；姓名监督残差虽救回32条历史共同错例，但误伤更多，强化约束后仍无净收益。姓名分支已按结果及用户反馈归档。当天一次线上v101为0.80874，最佳仍v96=0.81529；完整[当日结果与修复/误伤](reports/error_targeted_20261118/RESULTS.md)。

2026-11-17数据诊断：**不同模型的错误高度重叠，但没有证据证明80%是数据上限。** 在同一批6954条开发集折外预测上，五个代表家族共同错914人，占当时线上最佳v81本地错误的72.2%；777人存在分类分歧。仅对已有概率做正权重平均、仍用0.5分类阈值，不能修复这914个共同错误。完整[统计](reports/data_error_diagnosis_20261117/error_overlap.json)、[错误重叠图](reports/data_error_diagnosis_20261117/error_overlap.png)和[综合证据](reports/data_error_diagnosis_20261117/synthesis.json)已保存。

困难集中在Earth人群；其与G层高度重叠，不能视为两个独立原因。缺失值数量没有呈现明显的准确率递减关系；13个属性完全相同而标签不同的情况仅涉及5人，也不足以认定普遍错标。固定CatBoost配方的训练样本从3090增至4636时，三折均改善，整体从80.544%到81.579%。这说明该配方仍能从更多样本中受益，不代表新增数据必然带来同等收益。见[数据质量](reports/data_error_diagnosis_20261117/data_quality.json)与[学习曲线](reports/learning_curve_20261117/result.json)。

根据这些发现，已完成三个预先固定的实验；均属于复用开发集的探索，未通过各自继续条件，没有上传。

| 数据驱动改进 | 本地准确率 | 匹配对照 | 净多预测正确人数 |
|---|---:|---:|---:|
| 按HomePlanet学习Mitra/Kumo混合权重 | 81.766% | 常数权重81.881% | −8 |
| Earth/G层清醒乘客的舱位局部校准 | 81.694% | 原始Mitra 81.651% | +3 |
| 冷冻乘客消费及同组一致属性补缺 | 81.723% | 原始Mitra 81.651% | +5 |

分群混合采用严格的内外层拆分，避免用已间接见过外层标签的旧OOF训练路由器；[独立审查](reports/data_error_diagnosis_20261117/gating_feasibility.json)保留方法边界。发现某个群体难预测，并不自动意味着单独为它建规则就能改善泛化。

新增[共同错例专项分析](reports/error_rescue_20261117/ANALYSIS.md)：914条中704条来自地球，尤其地球休眠的300条错例中296条实际未传送；这个群体内v81准确率65.46%，仅略高于全部猜传送的65.28%。扩展到16个模型族代表后，298条共同错例曾被其他模型判对，但修复较多的方法同时造成更多误伤，不能直接按修复数混合。复用严格内层缓存的关系校正仅净增1条，无实质收益。所有数字是开发集诊断，不是Kaggle隐藏测试标签分析。 随后同条件单折15步对照：原13列82.1829%、人数版81.4495%、同行休眠版81.3201%；该折292条共同错例分别只判对1/1/2条，没有显示纠错突破。此单折不纳入完整三折排名。

## 当前实际成绩

当前最佳为 **v96 Mitra + 同行人数/同舱人数，Kaggle Public Score 0.81529（81.529%）**，提交 ID `58467382`，上传 commit `99b1902`。2026-11-18 10:06（北京时间）的官方账号最佳排名快照为 **109 / 2164，约前 5.037%**：前10%已达成，严格前5%尚未达到。滚动榜快照不是比赛最终名次；[当日官方记录](reports/proposal_ready_nonlinear_residual_v2/online_attempts/20261118T020611052500Z.json)保留证据。

| 方案 | 同一开发集随机三折 | Kaggle Public Score |
|---|---:|---:|
| v44 三种子树模型集成 | 81.98159% | 80.944% |
| v71 原始 Mitra-v2 | 81.65085% | 81.155% |
| v81 Mitra 全参数适配 | 81.79465% | 81.412% |
| v83 Kumo Medium8 | 81.83779% | 80.851% |
| v84 raw Mitra / TabDPT / Kumo 等权 | 81.57895% | 80.967% |
| v88 Mitra 双训练种子均值 | 81.89531% | 81.365% |
| EXAONE v91 默认8视图 | 80.98936% | 80.664% |
| EXAONE v92 / 原始 Mitra 各半 | 81.42077% | 未上传 |
| v93 Mitra support3072 | **82.11102%** | 81.295% |
| iLTM r128bn 原生8成员 | 未完成（按用户转向停止CV） | 78.138% |
| v95 消费差额、固定15步适配 | 未跑（直接线上） | 80.944% |
| v96 同行人数/同舱人数、固定15步 | 未跑（直接线上） | **81.529%** |
| v97 适配v81 / 树模型各半 | 未跑（缓存直接线上） | 81.155% |
| v98 新旧最佳v96 / v81各半 | 未跑（缓存直接线上） | 81.365% |
| v101 rawMitra + 非线性残差 | 81.89531% | 80.874% |

本地最高为 v93 的82.11102%，未达到历史83%门槛；线上最佳是v96，两者分开报告。v96上传时无新增本地评分，使用用户直接线上探索授权；原门槛失败和历史例外不改写。原始Mitra首次达到前10%的[独立复核](reports/online_batch_20261112/goal_verification.json)和[早期8个代表方案对照](reports/a_tier_batch_20261112/RESULTS.md)继续保留。

当前目标为线上前5%。本地差距在1个百分点以内按同一实用性能档看待，不能因为零点几的提升宣称稳定领先。**原先汇总的87个完整随机三折配置：A档64、B档16、C档7，S档仍为空。** 后续共同错例专项另表记录，未重建该历史分档。 新本地锚点上移导致部分边界配置机械换档，不表示它们突然变差。 这是一批具体配方的分档，不是整个模型族的结论；见[方案分级](reports/METHOD_TIERS.md)与[可重建明细](reports/method_tiers.csv)。分组实验单独记录，不混入随机三折排名。

## 后续候选与历史实验

新增待提交候选 **v99 同行休眠信息**：在v96基础上增加排除本人的休眠比例及已知同行数量，真实17列、固定15步全量训练，4277行保存重载差0。当前尚无线上成绩，因共同错例专项优先而保留待定；没有切换第二账号增加次数。详见[下一轮状态](reports/next_round_20261117/PLAN.md)。

以下为11月17日历史：该批9次直接线上测试全部完成，当日总计10次已用尽。v96成为新最佳：保留原13列编码，加入固定无标签名单计数，直接15步全量训练并导出，耗时346.22秒，省略50步逐次验证选点。当前无运行中训练，v99待提交。后续继续线上优先，以v96为基准，不恢复已停止的CV；+0.117个百分点只是本次线上观察，尚非稳定优势。

用户最新选择 **线上探索优先**：暂停额外随机/分组交叉验证，先用全部训练数据拟合、检查完整提交与保存重载，再直接线上评测。iLTM的FALSE版CV已按这一要求中断，尚无完整折分数；该中断不代表模型效果差。iLTM和EXAONE均已直接提交，分别78.138%、80.664%，未超过最佳，参见[新授权](reports/online_iteration_20261117/online_first_policy.json)。复用完整缓存的v84三结构等权已直接提交，得到80.967%，未超过最佳；参见[本批线上结果](reports/online_iteration_20261117/RESULTS.md)。

已覆盖逻辑回归、CatBoost/LightGBM、AutoGluon、MLP、TabM、FT-Transformer、TabR、SCARF，以及TabPFN、TabICL、两代Mitra、TabDPT、Kumo、EXAONE等预训练表格模型。FT-Transformer80.47%、TabR79.67%、SCARF79.80%（匹配scratch79.62%）是固定首轮配方的结果；不能凭一个配置否定整个模型族。早期方法的输入与评估边界见[方法比较](reports/method_comparison.json)，后续新增结果以各 `reports/run_*/` 及本轮提交记录为准。

v81训练全部75,670,026个参数，使用1024支持行/256查询行的训练episode，在完整内层上下文上从0至50步选择，再从原始权重重新拟合所选步数。最终全量模型选择15步，全部4277行保存重载一致。它曾取得当时最佳线上成绩；后续双种子、分组episode、原始/适配均值等探索均未超过它，之后v96人数特征版成为新最佳。历史较早的全参数内存失败保留，不代表当前实现仍不可运行。

EXAONE v91分组三折81.66523%，匹配原始Mitra81.69399%；固定各半v92分组81.86655%，但随机三折低于原始Mitra且三折均下降，因此未通过双划分继续条件，已归档、不上传、不继续搜索权重。新代TabPFN按用户决定关闭；Orion-BiX、TabForestPFN与TabFlex的交付/复现限制分别留有独立调查记录，不能把资源或来源问题说成模型准确率差。

固定 **v93 Mitra support3072** 已完成：保持v81每步相同的256行查询和原1024行支持样本，只追加2048行支持。三折82.11102%、logloss0.371853，较v81净多22正确，逐折+11/−2/+13，三折logloss均降低；选步为29/50/17。root独立复算通过，满足事前继续条件。首次全量导出在inner step1 validation触发MPS内存上限，旧失败保留；独立状态保持诊断通过后，只增加验证前缓存清理，完整50步选47并全量重训47，50份状态证明及4277行重载全部通过（差异0）。提交58454737得到0.81295，低于v81=0.81412，本地小幅领先未转化为线上收益。执行路径没有重跑完整三折；继承原run056算法证据，不新增一条CV成绩。仍小于1pp、复用开发集且计算量更大，不构成S档。[完整结果](reports/run_056/summary.json)、[事前协议](reports/mitra_support3072_screen_review.json)和[独立导出决定](reports/online_iteration_20261116/support3072_decision.json)均已保存。

方法覆盖复核与反方审查又提出v94：仅将Mitra的HomePlanet/Destination改为train-only one-hot，保留其余11列与原mirror。27项CPU/mock检查通过，但真实小型CPU/MPS数值检查未通过事前0.02容差，因此停止于资源阶段，没有准确率、未运行run057、没有上传。不能据此判定one-hot效果差；[覆盖复核](reports/remaining_method_coverage_review_20261117.json)、[反方报告](reports/remaining_method_coverage_battle_20261117.json)、[失败记录](reports/mitra_nominal_onehot_resources/result.json)保留。

后备TabDPT原生逐查询检索确实不同于已测完整上下文，但CPU依赖初始化独立复现重复OpenMP运行库错误，未进入模型或准确率实验；保持环境不变并归档，[故障证据](reports/tabdpt_retrieval_cpu_setup.json)。11月16日批次历史已用9/10次；11月17日13:03重新核验，今日已用1次、可用9次。按用户最新要求更新[今日计划](reports/online_iteration_20261117/PLAN.md)，未来最多新增9次，旧批剩余1次不叠加。当前仍未达到前5%。

iLTM新结构的两个合成CPU单成员案例已完整通过训练、预测、保存重载（概率差均0），共317.54秒；但事前25%余量下，完整8成员内存估算11.14/10.89GiB超10GiB，固定配置按原规则归档。没有实际8成员OOM或真实准确率，不能据此判定效果差；[资源结果及独立复核](reports/iltm_native_cpu1_resources/root_audit.json)。

后续独立立项的官方 **iLTM r128bn（128维瓶颈、无树特征）** 已通过原生完整8成员资源检查：8693条合成训练行、4277条查询，32分钟有效运行/46分钟墙钟，采样进程树RSS峰值6.08GiB，全部预测保存重载差0，顺序/子集最大差约6e−8且分类一致。仅资源资格通过，尚无真实准确率；不能把较小checkpoint视为效果更好。旧xgbrconcat失败记录不改。下一步依[固定CV协议](reports/iltm_r128bn_cv_design.json)运行原6954开发集三折，完整结果达到当前A档地板81.1110%才做原分组三折，两阶段通过也仅供另行评估探索价值。见[资源复核](reports/iltm_r128bn_full8_resources/root_audit.json)与[12项测试及独立代码审查](reports/iltm_r128bn_cv_static_review.json)。

首次真实CV在启动训练前被环境检查拦下，0折、无候选分数：`threadpoolctl`与`sklearn`导入时都会自动设置OpenMP重复运行库容忍开关。旧资源入口只检查变量不存在，不足以证明训练期间仍禁用；旧成功运行事实保留，但不能作为显式禁用开关后的资源资格。该旧CV路径已归档。独立显式`FALSE`迁移验证现已通过：完整新单成员训练与原概率差0，旧8成员在新环境中全4277预测和保存重载差0，顺序/子集差不超过6e−8；498.59秒、采样峰值6.193GiB。此验证没有重新训练8成员，只支持在完整硬限制下继续同配方CV，不能改写旧证据。新入口保持核心数据/评分及模型调用AST一致，并强制检查已提交的独立审核。见[导入边界诊断](reports/iltm_r128bn_import_boundary_diagnostic.json)、[迁移复核](reports/iltm_false_runtime_bridge/root_audit.json)及[新入口审查](reports/iltm_r128bn_cv_false_static_review.json)。

## 数据与线上成绩

- 官方比赛：https://www.kaggle.com/competitions/spaceship-titanic
- 本地原始数据最初取自公开 GitHub 镜像；`data/raw/manifest.json` 保留初始来源和哈希。后续官方 API 下载的核验结果见 `reports/official_data_verification.json`。
- 仅使用原始训练标签、无标签测试集与提交格式模板。未使用其他人的预测文件或测试答案。
- 本地交叉验证、留出集准确率不等于 Kaggle 得分，也不能推断确切名次。
- 本机已完成 Kaggle OAuth 登录。训练与生成提交文件不依赖账号；其他机器仍需自行授权，凭据不包含在仓库中。

## Kaggle 自动提交

当前协作约定：每轮报告本地结果并 commit/push。11月18日已按持续自主探索授权单独释放v101现成候选并完成一次上传；后续仍须现场核对官方次数、文件与来源，不能重用已耗尽的历史批次计数。2026-11-17用户已取消此前周额度保留要求；历史11月16日批次已用9/10次，新[当日授权](reports/online_iteration_20261117/authorization.json)最多允许未来9次尝试，旧剩余次数不叠加，实际可用次数在上传前重新查询。每个候选先冻结方案、完成验证与完整导出重载，再由主agent单独决定是否值得提交。上传前检查实际每日配额、文件去重并在Mac桌面提醒；网络状态不明时只查记录，不盲目重传。探索许可不代表S档稳定领先，历史83%门槛与例外记录不改写。连续四个候选未提升至少0.10个百分点时，启动调研和审查两个subagent交叉评议。

新授权范围内的线上提交使用说明“已获授权”的桌面提醒，无需再次确认；`remind_submission.applescript`保留供确实需要确认的其他场景使用。Git commit/push无需逐次确认。

官方 CLI 支持首次 OAuth 登录后持续使用本机凭据。账号注册、首次授权、接受比赛规则以及平台要求的身份验证需要账号本人完成；正常下载、提交、等待评分、查询结果不需要每次操作网页。

```bash
uv pip install --python .venv/bin/python -r requirements-kaggle.txt
.venv/bin/python kaggle_workflow.py login
.venv/bin/python kaggle_workflow.py download
.venv/bin/python kaggle_workflow.py check --run run_002
# 把下载核验记录和本次代码提交到 Git 后：
.venv/bin/python kaggle_workflow.py submit --run run_002 --wait 180
# 评分尚未完成时，只查询，不重新提交：
.venv/bin/python kaggle_workflow.py status --run run_002
```

首次还需要在 https://www.kaggle.com/competitions/spaceship-titanic 加入比赛。`download` 从官方 API 获取数据并核对现有镜像；全新 clone 会创建本地 CSV。现有文件不一致时停止，不覆盖旧实验数据。

下载逐个调用官方文件接口。2026-11-12 实测整包接口返回 403，而三个单文件接口均可正常下载；不能仅凭整包接口的通用错误提示断言用户未接受规则。

脚本在上传前检查 4,277 行、布尔标签和 ID 顺序，并记录当前 commit、文件 SHA-256、版本、提交 ID 和评测结果。结果位于 `reports/<run>/online_attempts/`，完成后也要 commit。原始命令输出仅写入本地忽略目录 `submission_logs/`。

上传会消耗赛事提交配额。脚本不会自动重试上传；即使请求中断，重复文件也会被拦截。先检查提交历史及本地日志，拿到 ID 后可运行 `status --run run_002 --ref <ID>`。确实没有上传成功时，手动核对并归档尝试记录后再重试。

每日配额、评分排队、账号权限和网络连通性受 Kaggle 控制。凭据保存在用户本机，不能提交进 GitHub。当前脚本由本机会话执行，没有配置定时任务或 GitHub Actions。

官方文档：https://www.kaggle.com/docs/api 、https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md

## 运行

```bash
cd data_analysis_hw
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python -r requirements-kaggle.txt
.venv/bin/python kaggle_workflow.py login
.venv/bin/python kaggle_workflow.py download
.venv/bin/python experiment.py init --run run_001
.venv/bin/python experiment.py develop --run run_001
.venv/bin/python experiment.py group-audit --run run_001
.venv/bin/python experiment.py finalize --run run_001
.venv/bin/python report.py --run run_001
```

`init` 只用于新实验，已有目录会拒绝覆盖。`develop` 可从已完成版本继续；`finalize` 拒绝重复打开已经评估过的留出集。运行新版本请保存新的版本号、配置和源代码，不覆盖旧结果。

上面的命令是从空目录复现的顺序，不要对已经存在的实验重复初始化。当前项目的 `run_001` 是第一轮六候选探索，`run_002` 在相同开发集追加三项实验；相同候选经配置、特征矩阵和划分一致性检查后复用 OOF。最终评估与提交文件仅生成在 `run_002`。两个 run 共享同一留出集，不能把另一 run 的重复评估称为新的独立验证。

## 验证协议

以下为初始实验协议。后续开发使用 `iterate.py`，候选配置、依赖及源代码哈希记录在每轮 `protocol.json`。加入的同行属性只来自无标签的 train+test 名单，不传播同行标签；实际 train/test 的同行组编号没有交集。各轮采用同一开发集，反复探索后的最高分仍属于开发结果，不能称作新的独立测试成绩。

1. 固定随机种子，将 20% 训练样本保留为最终留出集；剩余 80% 为开发集。
2. 所有候选在完全相同的三折分层交叉验证中比较，阈值固定 0.5。
3. 按开发集折外预测（OOF）准确率选择，准确率相同时参考 Log Loss。
4. 学习型预处理仅在训练折拟合。CatBoost 迭代数预先固定，不用外层验证标签提前停止。
5. 特征工程仅使用当前乘客这一行的属性，没有目标编码、同行标签传播或利用测试答案。GroupNumber 是乘客编号前缀，GroupMember 是后缀序号；不是“同行人数”。
6. 选定版本在开发集做同行群组不交叉的三折压力测试，检查对陌生群组的泛化；它不用于再次选模型。
7. 选定模型后只打开留出集一次。后续版本仍应依赖开发集交叉验证；不能把这一留出集持续当作未见数据使用。
8. 最终用全部有标签训练数据重训，生成 4,277 行布尔预测。

随机留出集允许同行乘客分布在两边，模拟随机隐藏乘客标签；群组压力测试回答不同问题，两者要一起报告。

## 版本

| 版本 | 方法 | 目的 |
|---|---|---|
| v0 | 原始特征 + 逻辑回归 | 建立简单可解释基线 |
| v1 | 原始特征 + CatBoost | 检查非线性模型带来的收益 |
| v2 | 舱位拆分、消费汇总/对数、缺失标记、乘客编号结构 + CatBoost | 检查整组特征工程的作用 |
| v3 | 相同特征，更浅树、更强正则 | 检查复杂度与泛化 |
| v4 | 相同特征，更深树 | 检查拟合能力 |
| v5 | v2/v3/v4 等权概率平均 | 检查简单融合是否有收益 |
| v6 | v2 移除乘客编号结构 | 检查 GroupNumber/GroupMember 的贡献 |
| v7 | v2 移除额外消费特征，保留原始消费金额 | 检查消费特征工程的贡献 |
| v8 | v2 加休眠/消费逻辑填补 | 检查规则填补的收益 |

v1/v2 是整组特征方案对照；v6/v7 分别做编号与额外消费特征消融。消融结果说明该验证设置下的预测贡献，不表示现实世界因果效应。

## 三人分工

- A：数据与缺失值分析，解释数据质量、划分方法。
- B：特征工程，解释设计依据和消融结果。
- C：模型与验证，解释 CV、融合、误差和线上结果。

## 主要输出

已完成实验的可审查摘要保存在 [`reports/run_002/REPORT.md`](reports/run_002/REPORT.md)，第一轮对照保存在 `reports/run_001/`。Git 记录代码与实验摘要；原始数据、虚拟环境、训练模型和中间缓存保留在本地，通过固定依赖与协议复现。

每轮位于 `artifacts/<run>/`，包含协议、源代码快照、固定划分、各版本 OOF、开发集结果、群组压力测试、留出集预测、模型、特征重要性、最终提交文件。

提交前先在 Kaggle 登录并加入比赛，上传 `submission.csv`。保留 Kaggle 的实际分数、名次、总队伍数、提交时间和版本号。不要在聊天或代码仓库粘贴 API 密钥。

早期树模型与Mitra固定各半集成未通过原稳定性资格；随后用户单独授权一次探索，线上80.804%，低于原始Mitra。完整历史见[验证结果](reports/top5_crossfamily_phase/RESULTS.md)及[线上记录](reports/crossfamily_online_20261113/RESULTS.md)。

100步最后模块适配的旧稳定性审核失败记录仍保留；本轮另做一次探索，线上81.155%，未超过v81。见[旧验证结果](reports/top5_adaptation_phase/RESULTS.md)及本轮逐次提交记录。

早期阶段结论见[历史前5%迭代记录](reports/top5_iteration_summary.md)；当前状态以[2026-11-16自主迭代记录](reports/online_iteration_20261116/RESULTS.md)及各版本独立协议、评估与提交记录为准。
