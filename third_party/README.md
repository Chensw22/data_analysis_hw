# TabPFN-v2 模型来源与许可

Built with PriorLabs-TabPFN

- 公共模型：https://huggingface.co/Prior-Labs/TabPFN-v2-clf
- 使用权重：`tabpfn-v2-classifier-v2_default.ckpt`，不使用其他人的竞赛预测或测试标签。
- 权重许可证：Prior Labs License v1.1 (May2025)，附于本目录 `tabpfn-v2-LICENSE.txt`。
- 来源：https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt
- 模型文件只保存在本地 `artifacts/pretrained/`；下载来源与哈希另行记录。
- 代码包版本：TabPFN 9.2.0；本实验显式指定 V2 和本地模型路径，不调用云推理服务。
- 独立环境：`uv venv --python 3.12 .venv-tabpfn`，`uv pip install --python .venv-tabpfn/bin/python -r requirements-tabpfn.txt`。
- 先执行 `.venv-tabpfn/bin/python tabpfn_candidate.py probe` 验证本机 MPS 兼容性。该探针不计算准确率，不用于满足83%门槛；正式评估使用同一固定开发集完整三折。
