# 小规模 MPS 兼容检查

仅128行合成支持数据/17行查询、1次排列、FP32，MPS内存上限0.10。与进行中的Mitra训练并行，故不作为性能基准，也不替代完整尺寸资源门槛。

调用已固定的`tabdpt_resource_probe.synthetic(128,17,20261116)`、`new_model("mps")`、`model.fit(x,y)`、`probabilities(model,z,ensembles=1)`，线程数4，120秒超时。全部17行概率有效，约6.53秒；记录的MPS值为运行后快照，不是峰值。未计算准确率。
