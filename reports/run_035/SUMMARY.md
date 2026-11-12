# FT-Transformer method screen

Official default architecture (3 blocks, 192 dimensions, 8 attention heads; 905,857 parameters in the resource probe) trained from scratch on compact13 plus seven categorical missing indicators. Same fixed three-fold development split and threshold0.5 as all previous candidates.

Accuracy **80.47167%**, log loss **0.404341**, 326.09seconds. Fold accuracy:80.37101%,81.01812%,80.02588%. The v6 control reproduces81.57895%; current strongest v44 pipeline is81.98159%. No online submission; the >83% gate fails.

Inner minimum-logloss epochs23/27/17 were selected, followed by fresh full-outertrain preprocessing and network refits. Early stopping occurred at63/67/57 epochs; none reached500. At the selected epochs, training BCE was0.3802/0.3738/0.3850 and validation logloss0.4094/0.3982/0.4045. Continuing training lowered training BCE to about0.295/0.289/0.295 while validation loss rose to0.466/0.460/0.450. This supports overfitting of this recipe, not a training-time shortage.

Decision: defer tuning this default recipe until the distinct TabR and SCARF screens finish. This result does not rule out all FT-Transformer architectures or preprocessing. Historical TabM shares the input/preprocessing contract, but the training recipes differ; v44 includes extra engineered/spatial information and is a practical pipeline reference, not a pure architecture comparison.
