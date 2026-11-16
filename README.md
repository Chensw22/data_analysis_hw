# Spaceship Titanic 课程项目

目标：三人团队稳妥完成可复现的竞赛项目，保留基线、改进、失败尝试和答辩证据。

## 当前实际成绩

当前最佳为 **v81 Mitra 全参数适配，Kaggle Public Score 0.81412（81.412%）**，提交 ID `58424738`，上传 commit `8e65a567eed7`。2026-11-17 04:52（北京时间）的官方账号最佳排名为 **114 / 2158，约前 5.283%**：前10%已达成，前5%尚未达成。滚动榜快照不是比赛最终名次；[官方快照](reports/online_iteration_20261116/post_exaone_screen_rank.json)与[本轮逐次提交记录](reports/online_iteration_20261116/RESULTS.md)保留完整证据。

| 方案 | 同一开发集随机三折 | Kaggle Public Score |
|---|---:|---:|
| v44 三种子树模型集成 | 81.98159% | 80.944% |
| v71 原始 Mitra-v2 | 81.65085% | 81.155% |
| v81 Mitra 全参数适配 | 81.79465% | **81.412%** |
| v83 Kumo Medium8 | 81.83779% | 80.851% |
| v88 Mitra 双训练种子均值 | 81.89531% | 81.365% |
| EXAONE v91 默认8视图 | 80.98936% | 未上传 |
| EXAONE v92 / 原始 Mitra 各半 | 81.42077% | 未上传 |

本地最高仍是 v44 的81.98159%，未达到历史83%门槛；线上最佳是v81，两者分开报告。新版本的探索提交使用后续明确授权，原门槛失败和历史例外不改写。原始Mitra首次达到前10%的[独立复核](reports/online_batch_20261112/goal_verification.json)和[早期8个代表方案对照](reports/a_tier_batch_20261112/RESULTS.md)继续保留。

当前目标为线上前5%。本地差距在1个百分点以内按同一实用性能档看待，不能因为零点几的提升宣称稳定领先。**86个完整随机三折配置：A档67、B档13、C档6，S档仍为空。** 这是一批具体配方的分档，不是整个模型族的结论；见[方案分级](reports/METHOD_TIERS.md)与[可重建明细](reports/method_tiers.csv)。分组实验单独记录，不混入随机三折排名。

## 当前实验方向

已覆盖逻辑回归、CatBoost/LightGBM、AutoGluon、MLP、TabM、FT-Transformer、TabR、SCARF，以及TabPFN、TabICL、两代Mitra、TabDPT、Kumo、EXAONE等预训练表格模型。FT-Transformer80.47%、TabR79.67%、SCARF79.80%（匹配scratch79.62%）是固定首轮配方的结果；不能凭一个配置否定整个模型族。早期方法的输入与评估边界见[方法比较](reports/method_comparison.json)，后续新增结果以各 `reports/run_*/` 及本轮提交记录为准。

v81训练全部75,670,026个参数，使用1024支持行/256查询行的训练episode，在完整内层上下文上从0至50步选择，再从原始权重重新拟合所选步数。最终全量模型选择15步，全部4277行保存重载一致。它取得了目前最佳线上成绩；后续双种子、分组episode、原始/适配均值等探索均未超过它。历史较早的全参数内存失败保留，不代表当前实现仍不可运行。

EXAONE v91分组三折81.66523%，匹配原始Mitra81.69399%；固定各半v92分组81.86655%，但随机三折低于原始Mitra且三折均下降，因此未通过双划分继续条件，已归档、不上传、不继续搜索权重。新代TabPFN按用户决定关闭；Orion-BiX、TabForestPFN与TabFlex的交付/复现限制分别留有独立调查记录，不能把资源或来源问题说成模型准确率差。

下一项为固定 **v93 Mitra support3072**：保持v81每步相同的256行查询和原1024行支持样本，只追加2048行支持。合成四次更新及两种完整内层验证已通过，正式三折上限固定5小时；这仍是待评估的机制假设，没有新准确率。[资源记录](reports/mitra_long_context_resources/result.json)与[事前筛选协议](reports/mitra_support3072_screen_review.json)分别记录资源资格和效果门槛。

## 数据与线上成绩

- 官方比赛：https://www.kaggle.com/competitions/spaceship-titanic
- 本地原始数据最初取自公开 GitHub 镜像；`data/raw/manifest.json` 保留初始来源和哈希。后续官方 API 下载的核验结果见 `reports/official_data_verification.json`。
- 仅使用原始训练标签、无标签测试集与提交格式模板。未使用其他人的预测文件或测试答案。
- 本地交叉验证、留出集准确率不等于 Kaggle 得分，也不能推断确切名次。
- 本机已完成 Kaggle OAuth 登录。训练与生成提交文件不依赖账号；其他机器仍需自行授权，凭据不包含在仓库中。

## Kaggle 自动提交

当前协作约定：每轮报告本地结果并 commit/push。2026-11-16最新[自主迭代授权](reports/online_iteration_20261116/authorization.json)允许有依据的探索性线上测试，本批最多10次，目前已用8次；本周累计使用上限70%，至少保留30%。每个候选先冻结方案、完成验证与完整导出重载，再由主agent单独决定是否值得提交。上传前检查实际每日配额、文件去重并在Mac桌面提醒；网络状态不明时只查记录，不盲目重传。探索许可不代表S档稳定领先，历史83%门槛与例外记录不改写。连续四个候选未提升至少0.10个百分点时，启动调研和审查两个subagent交叉评议。

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
