# 项目约定

- 仓库：`Chensw22/data_analysis_hw`；当前本地目录就是该仓库的工作副本。
- 用户已要求每次更新用 commit 管理；每个完成且验证过的逻辑改动提交清晰 commit，并推送到 origin。不得覆盖、强推或改写已有历史。
- 竞赛：Spaceship Titanic。三人课程项目，优先稳定、可复现、能解释的结果。
- 实验事实以 `reports/` 的结果与协议、`artifacts/` 的本地完整记录为准，不把本地分数称为 Kaggle 分数。
- 2026-11-12用户更新比较原则：本地准确率差距在1个百分点以内按同等实用性能看待，不因零点几的提升宣称胜出；仅有重复划分、配对比较及分组验证支持的一致实质提升才可称稳定更强。方案分档以同一个当前最高本地分数为锚点，A差距<=1pp，B为1–2pp，C为>2pp；这是筛选档位，不是统计显著性或整个模型族的评级，跨档边界也不能自动宣称更强。当前S档（稳定领先）空缺。完整档位见reports/method_tiers.csv与method_tiers.json，线上记录单独报告。
- run_001 与 run_002 使用同一个固定划分。run_002 已打开留出集，不能在后续迭代中把相同留出集重新声称为未见数据。
- 改进在开发集交叉验证中选择；公开榜仅作外部验证，不逐样本反推答案。
- Git 追踪代码、依赖、实验协议及汇总。凭据、原始数据、环境和模型缓存不得提交。Kaggle 认证使用用户本机的官方 CLI 配置。
- 上传 Kaggle 前核对目标竞赛、提交文件格式与版本 commit；保存提交 ID、分数和时间。不得因网络超时自动重复提交消耗配额。
- 不使用其他人的预测或公开测试答案进行训练/调参。
- 当前迭代目标已升级为线上进入前5%；前10%目标已达成。历史83%门槛及失败记录不得改写。新阶段按用户“稳定好的模型就提交”授权执行，先冻结稳定性审核协议并验证，再按新阶段记录判断候选；不因单次零点几的本地或线上增幅宣称稳定领先。
- 每轮本地结果主动告知用户，不因完成一轮而停止；每次完成的代码/结果修改 commit 并推送。
- 用户最新授权：新阶段出现经冻结协议审核为稳定好的候选时，可直接线上测试，无需逐次再次确认；该授权只适用于合格候选，不能盲目扫提交。上传前保存可审查的验证结果、候选文件哈希及commit，检查剩余次数并在Mac桌面弹提醒。既有`remind_submission.applescript`文案用于需要确认的场景；自动授权提交的提示应说明已获授权，不把关闭弹窗作为授权来源。Git commit/push无需逐次确认。
- 连续超过三轮提升不明显时，启动两个独立 subagent：一个调研改进路线，一个审查现有方案。让双方交叉质疑，由主 agent 核查结论再实施。将连续四个候选未带来至少 0.10 个百分点的开发集提升作为本次可操作的触发条件。
- 用户进一步允许子agent重新调研并修改整体方案，不局限传统大数据分析方法；委派代码需明确文件所有权并由主agent验证。
- 新增一次明确披露的group-primary验证修订：事实依据在reports/run_003/REVIEW.md已记录，不称新发现；既已观察部分旧group结果，故新划分仍属适应性开发验证。固定v74/v77两候选、v71/v44两参照，group seeds20261112/15/16；不得因结果换seed或删候选。完整政策与新授权分别为reports/top5_group_validation_review.json、reports/top5_group_authorization.json，旧random协议/授权/失败记录不改。全部固定队列完成且个体满足新A实验性测试资格后才可自动上传，S稳定领先另行判定；不把A资格表述为超过1pp的稳定优势。
- 用户最新设置周额度保留12%。每轮实验及新增子agent任务前，使用quota_guard.py读取当前会话最新weekly rate-limit元数据；remaining<=12%时停止新增工作、停止子agent并保存状态。元数据缺失/陈旧不能假装已知剩余额度。仅输出比例和时间，不提交个人额度状态、余额、凭据或会话日志。
- 2026-11-12用户明确“上传一次吧”，一次性授权proposal_run012低于83%的提交，已完成58265401（0.80967，225/2030快照）。这不是后续自动上传或永久下调门槛的授权。
- 2026-11-12用户另授权“给我三个模型尝试一下提交”：本批限v44三种子集成、TabPFN32、Mitra-v2直接推理，各一次上传；明确作为未达83%及稳定性门槛的批次例外，授权记录见reports/online_batch_20261112/authorization.json。该授权不延伸到第四次或后续批次。保留原CV门槛失败记录，不改写为通过。
- 本批三次授权已全部消耗：v44提交58275856得0.80944，TabPFN32提交58276170得0.80664，Mitra-v2提交58276251得0.81155。2026-11-12T14:44:08Z账号最佳排名142/2037（前6.97%），线上前10%目标达成；滚动榜随时间变化。证据以reports/online_batch_20261112/RESULTS.md及各提交online_attempts为准。不得继续沿用已消耗的批次上传授权。
- 用户随后要求“A的还有几个没跑，都跑一下”：按上一张A档代表方案表，授权尚未线上评测的TabPFN8、LightGBM线性叶+空间模型、TabICL工程特征版各一次提交，保留原配置并全数据拟合；这是新的三次批次例外，记录见reports/a_tier_batch_20261112/authorization.json，不扩展为51个A档微调/对照配置批量上传。
- A档补齐批次也已全部完成并消耗授权：LightGBM线性叶+空间模型58276576得0.80289，TabPFN8提交58276627得0.80687，TabICL提交58276759得0.80734。8个A档代表都有线上结果，Mitra仍最高0.81155；2026-11-12T15:03:57Z账号最佳142/2039。最新汇总reports/a_tier_batch_20261112/RESULTS.md；当时每日剩余2次不代表有继续上传授权。
- 2026-11-12用户质疑过早陷入微调，当前切换为方法级广度筛选：先完成FT-Transformer、TabR检索网络、SCARF自监督及匹配scratch对照，再统一决定值得调参的路线。冻结合理首轮recipe，未收敛或资源中断标记inconclusive；一次配置失败不能否定整个模型族。已有未运行的structure_counts保留代码但暂不启动，以免继续扩张小特征路线。

- 后续固定模型批次沿用group-primary门槛，不改写此前失败：v78/v79两代Mitra队列已全部完成均A/S=false；v80树模型与Mitra各半平均额外要求两次random不低于Mitra，完整六套OOF完成后仍A/S=false。证据分别见reports/top5_generation_phase和reports/top5_crossfamily_phase。这些划分均已使用，不能称独立验证。

- 既有v76/run042的100步适配也已完成单独group复核（reports/top5_adaptation_phase）：9折完整，group12持平、group15+4个正确、group16-11个正确，A/S=false，不自动上传、不延长到200步。mitra_full_episode_probe.py只是合成1024/256训练episode的全参资源探针；不构成真实候选训练/选步/线上资格，后续需要另行冻结实验。

- 用户2026-11-13明确“传一下看看吧”授权v80一次探索性例外，已完成58292572/0.80804，上传commit948a285，低于最佳Mitra0.81155；账号144/2038。授权已耗尽，证据reports/crossfamily_online_20261113/RESULTS.md，不能继续沿用。
- mitra_full_episode.py训练核心已实现并经CPU测试/审查，仍无正式screen wrapper、真实准确率或提交资格；小片段全参资源可行不等于效果改善。

- 2026-11-16用户恢复迭代：本周累计使用额度上限70%（保留30%），替代之前保留12%的设置；每轮及新委派前用quota_guard.py --reserve 30检查。用户允许主agent自主判断值得验证的版本并线上提交，本轮总上限10次，且受Kaggle实际每日可用次数限制。允许有依据的探索性上传，旧A/S失败记录保持不变，不把探索许可写为稳定领先。当前批次授权见reports/online_iteration_20261116/authorization.json；每份CSV最多尝试一次，不盲目重试，上传前桌面提醒、检查额度及完整导出重载。

- 2026-11-16新批第1次探索v74四视图已完成58411743/0.80921，低于最佳0.81155，账号150/2103；本批已用1/10。完整记录reports/proposal_nov16_mirror/online_attempts。v81全参50步run046已完整，三折81.79465%，相对Mitra+0.1438pp但逐折方向混合，A档且无稳定领先；TabDPT v1.3已通过完整MPS资源检查，下一步run047三折。

- 2026-11-16新批第2次探索v82 TabDPT已完成58423221/0.80687，未超过Mitra；本批2/10，账号151/2144。v81固定全参数配置正在全量拟合，决定文件reports/online_iteration_20261116/full_episode_decision.json允许一次探索。Kumo新架构仍在CPU环境/探针准备，尚无真实效果结论。

- 2026-11-16新批第3次v81全参数适配已完成58424738/0.81412，成为账号最佳；112/2146约前5.219%，目标前5%尚未完成。本批3/10。Kumo基础MPS资源检查因小案例分批概率差0.003741而失败，暂无正式screen；旧失败不得抹去。

- 本批第4次Kumo v83已完成58426834/0.80851；当前最佳v81=0.81412，113/2148约前5.26%，本批4/10。Kumo原MPS非连续out故障已局部修复并通过完整资源/重载，真实三折81.83779%；固定三架构均值v84三折81.57895%，暂无线上结果。下一步v76一次既定探索；新的group-disjoint适配仍在方案审核，不作已验证结论。

- v86固定rawMitra/v81各半已完成run050：81.80903%，仅比v81多1个正确且logloss稍差，仍A、无S；已冻结一次探索决定prior_adapt_decision。两个旧模型将先各自完整重载并验证组合，旧rawMitra未做重载的历史记录不改。group-disjoint全参v85代码/13tests/独立审查已完成并冻结review，等待串行GPU；仅group12三折vs同fold缓存rawMitra，禁止与v81随机CV宣称优势。结果独立group_episode_20261116目录。

- 新批第5次v76 lastblock100已完成58431795/0.81155，未超过v81=0.81412；账号114/2153，仍未前5%，本批5/10。全量100step选89、freshrefit89、全4277重载diff0。下一项v86固定各半均值，代码/decision均已commit，无需重训，但必须补两旧组件及新组合的全量重载。

- 新批第6次v86固定raw/adapt各半已完成58431944/0.81271，未超过v81=0.81412；账号114/2146，仍未前5%，本批6/10（剩4）。完整双组件及组合重载diff0，旧raw未验证记录保留。下一项v85固定group-episode50三折，review/code已commit8763c01，不扩大为6fold matched-v81比较，不混入随机CV评级。

- 第7次v85 group-episode已完成58436119/0.81248；matched groupCV81.56457% vsraw81.69399%（−9correct），三foldLL均改善但线上未超过v81=.81412。full6954/1739选31重训31、4277reload0。账号114/2151仍未前5%，本批7/10剩3。v85完整处置结束，后备run051固定seed+100以及两seed各半均值：v87solo只诊断，v88唯一主候选，review+code已commit；必须committed activation及新鲜官方排名才启动。均值acc>=v81、LL<=v81且至少2foldacc不降才继续fullfit；不追加seed/改权/上传solo。

- run051已完成并独立复算：v87solo81.65085%，v88固定两seed均值81.89531%/LL.374002，旧v81为81.79465%/.375093；净+7correct、2fold上升，固定continue通过，仍A/S=false。允许另commit探索decision后一次v88fullfit/export（newseed20261212、原full50、7200s），solo不上传、不加seed/改权。本批仍7/10。新v89原Mitra13+7数值missingflags/run052设计冻结reports/top5_missing_flags_review.json，需新完整resourceprobe，未打分，不改旧source。用户明确没有新代TabPFN授权并选择其他模型，该分支关闭。

- v89/run052已完整：81.63647% vsraw81.65085%，57修复/58损坏，LL.378945改善但2fold下降，固定gate失败；按预设归档，无fullfit/上传/追加适配/换flags/按missing子集事后拼接。资源两完整形状及重载全通过；root OOF复算一致。本批仍7/10，v88代码审查已通过，另commit seed_mean_decision后全量导出7200s。

- 第8次v88双训练种子均值已完成58439927/0.81365，未超过v81=.81412；新fullseed20261212内层选10并freshrefit10，旧v81不重训，两个组件及组合4277reload均0。账号114/2154约前5.2925%，目标仍未达。本批8/10剩2，不加seed/改权/上传诊断solo。

- Orion固定权重CPU核查与独立review发现实际standard RowInteraction、未启用BiAxial；此前按新多尺度结构的研究期待已更正。保存的step40不能推断总训练量，训练链未独立核实。不按此理由继续MPS/CV：v90/run053保留未运行，CPU事实与最小复现见reports/orion_bix_setup。未完成兼容草稿删除。下一路线为EXAONE CAST可行性审查，保留8/10上传状态。

- EXAONE CAST v91/run054已完成：80.98936%，较v44固定锚点低0.99223pp，刚进入A带；较v81少56correct，无稳定优势。完整MPS8693/4277及重载diff0，root复算一致。按事前protocol继续保存的group12三折；只通过后再另审探索上传，本批仍8/10。原model/core/资源配方冻结，不改seed/views/阈值。

- EXAONE saved group12三折完成81.66523%/LL.380938；matched raw81.69399%/.384031（172修复174损坏，净-2），v44为81.44953%。固定group门槛通过，仍无S。root已复算OOF；下一步另审一次探索decision，未批准前不得fullfit/线上。本批8/10不变。

- EXAONE_v92_rawMitra_half固定各半OOF双阶段完成：random81.42077%（相对raw少16correct、三fold全下降），group81.86655%（多12correct、3fold不降），LL均改善但双阶段gate=false。按预设归档，不导出/上传/搜权重/自动solo兜底。root复算缓存一致，未测试solo导出草稿删除，本批仍8/10，剩2。

- 下一项仅资源实验：固定Mitra3072支持/256查询，query和原1024支持前缀与v81逐step一致，只追加2048支持；保持既有双checkpoint/FP32+bf16/optimizer/backend。design见mitra_long_context_resource_design，四次persistent AdamW更新中插入3708/928及6954/1739验证；父进程监督1200s，13CPU/mock及独立审查通过。只有资源和预设成本gate通过后才能另commit是否运行正式三折；当前未授权CV/fullfit，失败不降支持大小或改backend救援。

- 资源实验随后完整通过（reports/mitra_long_context_resources，commit166e560），root已在任何新准确率前冻结正式v93/run056三折方案（reports/mitra_support3072_screen_review.json，e898b97），代码bd3a8af及17项CPU/mock检查通过。固定总时限18000s；仅与同划分缓存v81比较，完整三折acc不低、LL不高、至少2折acc不降且至少一个总指标严格改善才允许另审导出。失败归档，不追加支持大小/步数/seed/融合。当前第一折81.92407% vs v81 81.44953%，净+11correct，选29并重训29（4461152）；其余折仍在运行，不是完整候选结论。条件导出代码4eb9d10及13项检查已通过，但必须完整gate通过、root单独commit决定后才能读取全量标签；固定7200s及完整4277重载检查，尚未执行。线上本批仍8/10，最佳v81=.81412，最近快照114/2158（2026-11-16T20:52:08Z），前5%目标尚未达。

- v93/run056完整并root独立复算：82.11102%/LL.371853 vs v81 81.79465%/.375093，138修复116损坏，fold净+11/-2/+13，选步29/50/17，全部固定continue通过。结果08c084c；另commit41bd25b的support3072_decision批准一次7200s全量导出探索，当前正在执行，尚未上传；同1pp实用带、S=false。当前完整random87配方A64/B16/C7，锚点82.11102%。

- 方法覆盖审查先认为暂无新候选，反方发现原Mitra无序类别ordinal几何是具体遗漏；两方修订为固定v94 raw nominal one-hot一次对照，报告和计划978f451，代码e06cc5e/27CPU检查通过。但真实tiny CPU/MPS比较未满足固定0.02概率容差，资源failed_inconclusive，e13ef7e保留证据；未运行run057/真实准确率，不修改容差/编码/精度救援，不追加适配或线上。该故障与v93无关。
