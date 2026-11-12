# run_014：公开合成预训练 TabICLv2

固定tabicl2.2.0与110MB公开权重（SHA256见清单），本机MPS FP32，4 estimators、batch1。紧凑13特征80.97498%，完整工程特征81.19068%，紧凑模型与v25各50%为81.63647%。未超过v25或run012最佳81.92407%，未通过83%门槛。

模型编码、缺失填补、归一化均只在outertrain拟合；checkpoint每次fit校验哈希，禁用自动下载，数据不上传云端。预测后MPS驱动分配约2.95GB（紧凑）或4.97GB（完整特征），这不是全程峰值测量。全部折完成，未触及旧留出集或线上提交。

模型及许可来源：https://huggingface.co/jingang/TabICL 。预训练数据全部合成的声明见作者论文：https://arxiv.org/html/2602.11139v2 。
