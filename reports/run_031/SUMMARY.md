# 第31轮：严格嵌套的全局与休眠条件融合

v61全局两logit融合81.8666%，v62增加单个休眠可靠性系数81.7515%；均低于v44 81.9816%。对应logloss .376471/.376676仍好于单Cat，但本赛按accuracy，未达到83%门槛、无线上操作。

每outertrain内inner3重训v44/v54得到每行一次OOF，再拟合StandardScaler+L2LogReg C=.1。global输入两logit；conditional仅增加signedCryo*(logit54-logit44)，未知Cryo强制精确global回退。最后双基础模型在全outertrain重新训练，才评估外层；不使用既有全dev OOF拟合meta。

12组paired fits共享给两meta，缓存含run/source/config/seed/有序索引/y及对应特征值指纹。7项测试通过，真实单外折probe复现旧底座、61未知行bitwise fallback，无accuracy。首次正式screen前根据186.58秒probe及完整confirm工作量，固定阶段预算screen900/confirm2700/export1200秒；没有在看到候选成绩后增加预算。实际screen主耗时509秒。CSV中conditional约.044秒是复用已训练基础模型和meta后的增量时间，不代表独立训练只需.044秒。

inner基础模型约3090行、outer约4636行，学习曲线与校准漂移仍可能影响meta泛化。此方案失败后不继续扩Planet条件、C值或路由网格。
