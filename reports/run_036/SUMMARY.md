# TabR-S method screen

The official TabR-S core was adapted from pinned MIT-licensed source to exact PyTorch neighbor retrieval on MPS. No pretrained weights, public predictions or query labels were used. The differentiable label/key-difference aggregation was retained; each training query excludes itself by row identity. Independent code review and6 CPU tests passed.

Fixed three-fold development accuracy **79.66638%**, log loss **0.418900**, time179.30seconds. Fold accuracy79.98274%,79.76704%,79.24935%. The matched v6 control reproduces81.57895%; v44 remains81.98159%. The >83% gate fails; no online submission.

Inner best epochs5/7/6 were selected and freshly refit on all4636 outertrain rows. Early stopping at21/23/22 epochs, below the200 cap. Training losses continued to fall while inner validation worsened from0.440/0.399/0.400 to0.464/0.412/0.440. Increasing wall time alone is not supported by these curves.

Decision: defer tuning this fixed TabR-S recipe and finish SCARF versus its matched scratch control. This result does not reject all retrieval networks, feature encodings or TabR configurations.
