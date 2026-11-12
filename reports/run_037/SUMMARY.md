# SCARF matched mechanism comparison

On the unchanged fixed three-fold development split, the matched scratch MLP scores **79.62324%** (logloss0.429392,31.00seconds), while SCARF scores **79.79580%** (logloss0.426296,288.15seconds). Pretraining adds12 correct predictions out of6954, or0.17256 percentage points. Fold gains are+0.73339,-0.38827,+0.17256 percentage points: the sign is not consistent. The current best v44 control is rerun and exactly reproduces **81.98159%**.

The two neural candidates share architecture, input encoding, initial classifier-head seed and supervised-training protocol. SCARF alone adds unlabeled contrastive pretraining. Each fold performs three fresh pretraining phases and two supervised phases; only applicable training attributes plus actual test attributes enter pretraining. Validation labels and the old holdout never enter training or the pretraining pool.

Selected pretraining epochs25/42/62 and supervised epochs8/7/10. Pretraining searches stop at35/52/72 and supervised searches at48/47/50; all stop by patience, not the1000/500 epoch caps. The fixed corruption validation set is reused throughout each pretraining search.

Decision: this recipe shows a small, fold-inconsistent advantage over its matched scratch control, but remains2.18579 percentage points below v44. Do not prioritize tuning it before investigating a genuinely new pretrained model generation. This does not refute self-supervised learning in general. No online submission; all candidates remain below the >83% gate.
