"""Generate a report and standalone figures from completed experiment evidence."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main(run):
    out = ROOT/'artifacts'/run
    results = pd.read_csv(out/'development_results.csv')
    hold = json.loads((out/'holdout_metrics.json').read_text())
    group = json.loads((out/'group_audit.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results.sort_values('version')
    bars = ax.bar(ordered.version, ordered.accuracy*100, color=['#0d9488' if n==selection['version'] else '#2563eb' for n in ordered.version])
    ax.bar_label(bars, fmt='%.2f%%', padding=3)
    ax.set_ylim(max(0, ordered.accuracy.min()*100-3), min(100, ordered.accuracy.max()*100+2))
    ax.set_ylabel('Development out-of-fold accuracy (%)')
    ax.set_title('Fixed 3-fold validation; threshold = 0.5')
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout(); fig.savefig(out/'version_comparison.png', dpi=170); plt.close(fig)
    pred = pd.read_csv(out/'holdout_predictions.csv')
    raw = pd.read_csv(ROOT/'data/raw/train.csv')
    merged = pred.merge(raw.drop(columns='Transported'), on='PassengerId', validate='one_to_one')
    merged['correct'] = merged.target == merged.prediction.astype(int)
    breakdown = merged.groupby('HomePlanet', dropna=False).agg(n=('correct','size'), accuracy=('correct','mean'))
    breakdown.to_csv(out/'holdout_errors_by_planet.csv')
    # Wilson interval is descriptive under an IID approximation; passenger dependence can widen it.
    p, n, z = hold['accuracy'], hold['n'], 1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    lines = [f'# Spaceship Titanic：{run} 实验结果', '',
        '**以下均为本地实验结果。实际 Kaggle 评分和排名另见同目录 ONLINE_RESULT.md（如有），不能用本地准确率替代。**', '',
        f"开发集 {protocol['development_rows']} 人，留出集 {protocol['holdout_rows']} 人；固定三折、阈值 0.5。", '',
        '| 版本 | 开发集 OOF 准确率 | 折间标准差 | Log Loss | 训练耗时（秒） |',
        '|---|---:|---:|---:|---:|']
    for row in ordered.itertuples():
        duration = '复用三个模型预测' if row.version=='v5_equal_blend' else f'{row.seconds:.1f}'
        lines.append(f'| {row.version} | {row.accuracy:.4%} | {row.fold_std:.4%} | {row.log_loss:.5f} | {duration} |')
    lines += ['', f"选定版本：**{selection['version']}**；选择仅依据开发集，留出集没有参与选型。", '',
        f"- 一次性留出集准确率：**{p:.4%}**，Log Loss：{hold['log_loss']:.5f}。",
        f'- IID 近似下的 Wilson 95% 区间：{center-half:.2%}–{center+half:.2%}；同行相关性未计入，不是排名预测区间。',
        f"- 开发集群组隔离压力测试：**{group['accuracy']:.4%}**，三个折为 {', '.join(f'{v:.2%}' for v in group['fold_accuracy'])}。",
        f"- 留出集混淆矩阵（行是真值、列是预测，顺序 False/True）：`{hold['confusion_matrix']}`。", '',
        '![版本比较](version_comparison.png)', '',
        '## 如何解释结果', '',
        '- 比较版本时看同一开发集的 OOF 指标；留出集是另一批样本，不能用它与基线 CV 的差值宣称改进。',
        '- 最优 CV 是多个候选中的胜者，可能有选择偏差；一次性留出评估用于独立核对。',
        '- 树更深、模型融合不一定提升。保留没有获胜的版本作为实验材料。',
        '- v6/v7 相对于 v2 分别移除编号特征和额外消费特征；原始五项消费金额仍保留。v8 测试休眠与消费逻辑的缺失填补。',
        '- 原始数据最初来自公开镜像；官方数据的后续核验记录见项目 reports/official_data_verification.json（如有）。',
        '- 模型基于所有训练标签重训；提交文件通过列名、行数、ID 顺序和无缺失校验。', '',
        '## 下一步', '',
        '线上提交使用 artifacts 下本轮的 `submission.csv`，实际结果另行保存。后续改进继续使用开发集验证，不能依据已打开的留出集反复调参。', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out/'REPORT.md')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',default='run_001')
    main(parser.parse_args().run)
