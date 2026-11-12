# 第32轮：固定标签平滑

v64开发三折81.4639%，匹配CrossEntropy硬标签v63和原v34均81.9241%；v6 81.5789%。无改进，当前最佳仍v44 81.9816%，未达83%，无线上操作。第31轮计算量较大，编号不代表完成顺序。

固定epsilon=.1，把0/1训练标签改为.05/.95，保持v34特征/700树等参数；原hard-label空间专家不变。全局Cat输出q先作clip((q-.05)/.9,1e-6,1-1e-6)，再按原权重混合空间概率。反变换仅源于理想条件期望关系，有限模型误差放大约11.1%且可能饱和，不能称为校准保证；这是平滑训练加固定反变换的整体方案，不扫描epsilon或阈值。

600行功能probe不计算accuracy，软目标/逆变换数值关系及模型joblib往返通过。初始测试使用浮点精确等号比较.9+.05与.95导致断言失败，改为1e-15容差后通过；模型预测序列化仍要求逐值相等。实际get_all_params比较仅class_names、loss_function、eval_metric不同，其他自动默认一致。

实现依据：[CatBoost软标签含义](https://catboost.ai/docs/en/concepts/python-reference_pool_get_label)。标签平滑的通用研究动机来自[NeurIPS2019论文](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)，其神经网络结论不是本赛CatBoost提分保证。
