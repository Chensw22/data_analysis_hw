# 32-view TabPFN comparison

Same fixed development three-fold split, compact13 inputs and pretrained checkpoint as the eight-view control. Thirty-two independently generated views score **81.76589%**, versus eight views **81.55019%** (+0.21570 percentage points). Log loss improves from 0.380791 to 0.379566. The current v44 development best remains 81.98159%; the >83% online gate fails.

The 32-view candidate takes 561.95 seconds for all three folds; matched eight-view control takes 158.34 seconds. Increasing `n_estimators` changes the generated feature/class permutations; the eight-view run is not a cached prefix of the 32-view run. All resolved views and checkpoint hashes are recorded. This is a completed capacity/inference-budget comparison, not evidence that the entire neural-model family has reached its limit.

No online submission was made.
