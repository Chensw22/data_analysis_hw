# Mitra-v2 generation screen

The public75.7M-parameter synthetic-pretrained Mitra-v2 model was evaluated without task-specific gradient adaptation. Complete outer-training context, train-only compact13 preprocessing, fixed query batch2318, FP32 weights/input and MPS bf16 AMP. Independent attention-batch decomposition preserves full keys/softmax; CPU output and full-model gradient equivalence tests passed.

Fixed development3-fold accuracy **81.65085%**, logloss **0.380196**, time82.32seconds. Fold accuracy81.79465%,81.96721%,81.19068%. Baseline v6 exactly reproduces81.57895%; current best v44 remains81.98159%, and the best evaluated neural recipe TabPFN32 is81.76589%. The >83% gate fails; no online submission.

Prescore resource tests validated full4636 and8693-row inference without truncation. Query batches2318 exactly matched a common128-row subset in those synthetic MPS tests. Full representative adaptation attempts exceeded the declared65% MPS allocator limit, including with mathematically equivalent checkpointed attention chunks. Adaptation remains **unevaluated**, not an accuracy failure. The official50-step/eight-bag CUDA benchmark is not reproduced or claimed here.

Actual dev4636/128 pipeline probe validated finite outputs and original missing-sentinel encoding without computing accuracy. Per-fit logs, checkpoint SHA, dependency/source snapshots and runtime approval are retained. No previous holdout labels were used for training; full8693 resource probes used synthetic attributes and labels.

Next decision should compare strong-model error complementarity and a genuinely systematic ensemble protocol; this single zero-shot result does not rule out Mitra adaptation or the model family.
