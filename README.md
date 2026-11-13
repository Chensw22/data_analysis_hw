# Spaceship Titanic 课程项目

目标：三人团队稳妥完成可复现的竞赛项目，保留基线、改进、失败尝试和答辩证据。

## 当前实际成绩

2026-11-12 23:04（北京时间），分档表中的 **8个A档代表方案全部完成线上评测**。**Mitra-v2 直接推理**仍以 Kaggle Public Score **0.81155（81.155%）**最高，账号最佳排名 **142 / 2,039，约前 6.96%**，已达到线上前10%目标。这是滚动榜快照，不是永久最终名次。提交ID为 `58276251`，对应上传commit `2bc2b65`。参见[8个方案完整对照](reports/a_tier_batch_20261112/RESULTS.md)；[首次达到前10%的独立排行榜复核](reports/online_batch_20261112/goal_verification.json)保留。

线上对照：v44三种子集成80.944%，TabPFN32为80.664%；本次补齐TabPFN8为80.687%、LightGBM线性叶+空间模型80.289%、TabICL为80.734%。此前v34为80.967%（当时225/2030），初始v6为80.687%（当时517/2032）；Mitra相对此前最佳提高0.188个百分点。各次提交均保留独立ID、文件哈希与commit。8个代表的线上分数跨度仅0.866个百分点，当前仍无稳定领先的S档方案。

初始v6开发集三折OOF为81.58%，当时一次性留出评估为81.20%。Mitra本地三折为81.65%，线上为81.155%；本地与线上分数分别报告。本批由用户明确授权作为83%本地门槛的例外，原门槛失败记录未改写。

后续本地迭代目前最高为 `run_019 / v44_seed_ensemble` 的 **81.98%**（相同开发集三折 OOF），尚未达到83%；已完成新阶段重复划分及分组参照审核，详见[参照记录](reports/top5_reference_audit)。已上线的v34本地为81.92%，两次新随机划分为81.48%、81.62%，均好于原基线；同行隔离为81.39%，比基线低0.22个百分点，稳定性结论混合。旧留出集未再次用于选模型。每轮完整结果和失败方案保存在对应 `reports/` 子目录。汇总见 [迭代数据](reports/local_iterations.csv) 和 [进度图](reports/iteration_progress.png)，可运行 `summarize_iterations.py` 更新。

当前目标已升级为线上前5%，尚未达成。新阶段按冻结的[稳定性审核协议](reports/top5_validation_review.json)对合格候选自动测试，保留原83%门槛及历史例外记录；周额度至少保留12%。

## 当前实验方向

按用户最新原则，本地差距1个百分点以内按同等实用性能处理；当前没有已证实稳定领先的S档方案。76个已测配置的分档与证据见[方案分级](reports/METHOD_TIERS.md)，可运行 `classify_methods.py` 重建逐版本档位。

已尝试 MLP、TabM、TabPFN（含权重微调）、TabICL。最新方法级筛选完成：FT-Transformer80.47%、TabR79.67%、SCARF79.80%（匹配scratch对照79.62%）；均未超过v44。训练曲线均正常早停，不能简单归因于训练时间不足，也不能凭一个配置否定整个模型族。

Mitra-v2合成预训练模型已完成直接推理筛选：81.65%；TabPFN的32次集成得到81.77%，仍未超过v44的81.98%。完整8693行Mitra推理已通过资源检查。全参数适配曾因内存失败；后来采用冻结前11层、只训练最后模块和输出层，已通过全量资源及重载等价检查，完成内层选步与fresh refit的三折实验：81.52%，未超过直接推理；两折选中50步上限；随后唯一一次100步扩展得到81.65%，与直接推理准确率相同，未继续加长。固定四视图均值为81.58%，两次重复及分组方向为正，但随机重复平均提升仅0.086个百分点，未满足预设0.10个百分点测试门槛，因此未上传。四视图后来在固定分组复核的新两组中多对1个、少对1个，平均无增益；同行属性context20三组均下降，均未提交。Mitra-v1首轮79.95%，v1/v2固定均值81.16%，分组复核全部未通过。详见[四视图](reports/run_040/RESULTS.md)及[最后模块微调](reports/run_041/RESULTS.md)。包含原特征的AutoGluon非线性多层bagging已完成72个子模型，外层准确率80.73%，未提升。用户另授权v44、TabPFN32、Mitra-v2各一次线上对照，保留原83%门槛未通过的事实。方法覆盖、各自输入与实际成绩见[方法比较表](reports/method_comparison.csv)与[评估边界](reports/method_comparison.json)，运行 `summarize_methods.py` 可更新。

## 数据与线上成绩

- 官方比赛：https://www.kaggle.com/competitions/spaceship-titanic
- 本地原始数据最初取自公开 GitHub 镜像；`data/raw/manifest.json` 保留初始来源和哈希。后续官方 API 下载的核验结果见 `reports/official_data_verification.json`。
- 仅使用原始训练标签、无标签测试集与提交格式模板。未使用其他人的预测文件或测试答案。
- 本地交叉验证、留出集准确率不等于 Kaggle 得分，也不能推断确切名次。
- 本机已完成 Kaggle OAuth 登录。训练与生成提交文件不依赖账号；其他机器仍需自行授权，凭据不包含在仓库中。

## Kaggle 自动提交

当前协作约定：每轮先报告本地结果并 commit/push。用户最新目标为线上前5%，授权经预先冻结协议审核为稳定合格的候选自动提交；执行前核对官方剩余次数并在Mac桌面提醒。新阶段授权见[记录](reports/top5_authorization.json)，A档实验性测试资格与S档稳定领先声明分别判断，不能用本地分数替代线上排名。历史83%门槛及例外提交记录不改写。连续四个候选未提升至少0.10个百分点时，启动调研与审查两个subagent交叉评议。

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

树模型与Mitra固定各半集成本地81.97%，两次随机验证为81.67%、81.67%，但分组15/16相对Mitra一降一升、平均未提升，因此未取得线上测试资格，详见[结果](reports/top5_crossfamily_phase/RESULTS.md)。

100步最后模块适配已完成三组共9折复核：81.694%、81.507%、81.205%，相对原版为持平、+0.058、-0.158个百分点；未取得新线上测试资格，详见[完整结果](reports/top5_adaptation_phase/RESULTS.md)。
