# 第27轮：正则化线性叶节点

v56严格内层选择后的固定外三折准确率81.2482%，匹配constant-leaf控制v55为81.0181%。原v34对照81.9241%，v6为81.5789%，均精确复现；当前最佳v44 81.9816%。该模型族内有增益，但未超过最佳、未达到83%门槛，无线上操作。

沿用run005 LGB15预算和正则，固定700树。每个outertrain的inner3只选择constant、linear_lambda=1/10/100四个完整空间pipeline，按pooled accuracy、logloss、固定index择优。每个innertrain独立fit数值scaler、类别schema和空间专家；最终在全outertrain重新拟合。

8项测试及实际资源probe通过：模型dump确有非空linear leaf features，joblib逐值复现；1200秒阶段预算或任何搜索失败均不接受部分分数。内层选择总约20秒。单调性并未被约束，叶模型改变为线性函数；类别仍只用于分裂。参考[LightGBM官方文档](https://lightgbm.readthedocs.io/en/latest/Parameters.html)。
