# run_015：交叉拟合的消费投影

两条五变量逻辑回归分别使用原始消费和log1p消费；仅innertrain拟合median、StandardScaler和L2 C1分类器。CatBoost训练新增列来自inner3fold OOF，outerval来自fulloutertrain重新拟合投影。模型参数与原基线保持一致。

投影+CatBoost81.23382%，投影+空间专家81.43515%，未超过原基线或最佳81.92407%，未通过83%门槛，未提交线上。

评议分歧：低维outertrain监督投影直接在训练集变换也属于有效外层评估；crossfit能减少自身标签乐观，但会引入不同折的映射差异。主Agent预先冻结crossfit版本，未在两实现之间择优。双方代码审查未发现外层目标泄漏。
