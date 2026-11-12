# run_005：模型家族对照

LightGBM、RBF SVM、MLP均未超过已有CatBoost基线81.5789%，候选未达到83%门槛，未提交线上。详细分数见screen_results.csv。

所有模型使用相同开发集和外层折；SVM/MLP的填补、缩放只在训练折拟合，SVM概率标定和MLP提前停止均不使用外层验证标签。LightGBM依赖见requirements-models.txt；macOS使用sklearn自带libomp且不修改系统库。
