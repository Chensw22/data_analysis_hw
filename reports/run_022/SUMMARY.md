# 第22轮：XGBoost独立模型与固定融合

固定三折开发集准确率：v49 XGBoost 80.8456%；v50与v34各50%融合81.6796%。同时复现原始基线81.5789%与v34 81.9241%。当前最佳v44 81.9816%，本轮无改进，严格83%门槛未通过，无线上提交。

XGBoost3.4.1，固定depth4、lr.035、min_child_weight5、L2=5、L1=.1、row subsample.85、column subsample.9；最多1800轮，训练折内20% logloss早停60轮，再新建预处理器/模型在全部外层训练数据上拟合选中轮数。分类词表仅在对应训练数据拟合，dense onehot避免稀疏零被解释为缺失；金额NaN交给模型处理。使用原有v34消费对比，融合权重预先固定，未用外层标签调参。

400行功能探针（未计算accuracy）验证未知分类取值预测有限、joblib重载预测最大差0；基线准确率与历史精确相同。参数含义来源：[XGBoost官方参数文档](https://xgboost.readthedocs.io/en/stable/parameter.html)。
