# 本地预训练权重与乘客验证记录

在项目的 artifacts/pretrained、artifacts/limix2_setup 和 artifacts/nov23_tabswift 核对了本地权重，并检查实际乘客评分。下表的历史记录采用了不同的划分和模型配方。公开榜成绩仅表示对应提交的成绩。

| 缓存的模型 | 已有乘客证据 |
|---|---|
| Mitra v1、v2 | v1 零样本历史筛选为 79.95%，v2 有完整按订票组划分的 81.69%。[v1 筛选](../run_044/screen_results.csv) · [v2 分组记录](../top5_reference_audit/mitra/summary.json) |
| TabPFN v2 | 默认检查点以及三个不同架构的旧版权重均纳入过实际筛选。[多架构结果](../run_026/screen_results.csv) |
| TabICL v2、TabSwift | TabICL 已有折外预测和微调实测；TabSwift 的三次完整按预订划分分别达到 81.21%、81.38%、80.90%。[TabICL](../nov23_tabicl_finetune/result.json) · [TabSwift](../nov23_tabswift/RESULTS.md) |
| EXAONE | 有完整分组预测和 81.67% 的已存准确率。[分组验证](../exaone_group_20261117/summary.json) |
| Kumo、TabDPT v1.3 | 各自有真实公开榜回执，相关提交分别为 80.851%、80.687%。[Kumo](../proposal_nov16_kumo/online_attempts/20261116T102257033351Z.json) · [TabDPT](../proposal_nov16_tabdpt/online_attempts/20261116T082711151879Z.json) |
| LimiX-2 | 两个三折 CPU 试验均生成 6,954 人的预测，分别为 73.94%、77.62%。[乘客配置](../limix2_passenger_trial/result.json) · [出发地配置](../limix2_planet_trial/result.json) |
| iLTM r128bn | 对 8,693 人训练，生成测试集全部 4,277 行预测，已有公开榜回执。[导出记录](../proposal_online_first_iltm/manifest.json) |
| iLTM xgbrconcat | 合成数据的单成员 4,636 行训练可完成，实测进程最高内存 5.75 GB。旧报告是对八成员内存的外推，单成员在实际乘客上的效果仍待检验。[实测及预测区分](../iltm_native_cpu1_resources/root_audit.json) |
| Orion-BiX v1.1 | 以前仅凭 64 个合成训练例、17 个合成查询检查原生 CPU 概率及 tensor，直到本次才生成真实乘客评分。[旧运行](../orion_bix_setup/native_cpu_verified/result.json) · [旧结论](../orion_bix_setup/closure.json) |

Orion 作者的权重模型卡和项目代码均标注 MIT。实际加载的权重使用标准行注意力，旧原生探针逐个 tensor 比对了形状和值，并确认没有实例化双轴模块。本次直接使用同一提交的原生分类器及同一权重，每次启动均重新验证供应商文件及权重摘要。[权重模型卡](https://huggingface.co/Lexsi/Orion-BiX) · [代码与许可](https://github.com/Lexsi-Labs/Orion-BiX/blob/060d34606f56c591c47e0b6cbe971e5f5035f33e/LICENSE)
