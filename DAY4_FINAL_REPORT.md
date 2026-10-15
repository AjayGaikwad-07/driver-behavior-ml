# Day 4 — Feature Ablation and Unseen-Driver Evaluation

## Objective
Compare Random Forest and ExtraTrees using motion-only, state-only, and combined features for nine driver-behavior classes.

## Dataset and evaluation
- Training rows: 407,462
- Test rows: 173,344
- Training drivers: 393
- Unseen test drivers: 98
- Driver overlap: 0
- Evaluation: one held-out-driver split

## Results

| Model | Feature group | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Train time (s) |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | Combined | 0.2665 | 0.2689 | 0.1876 | 0.2184 | 25.47 |
| ExtraTrees | Combined | 0.2676 | 0.2645 | 0.1801 | 0.1924 | 12.55 |
| Random Forest | Motion only | 0.2486 | 0.2625 | 0.1758 | 0.2010 | 22.55 |
| ExtraTrees | Motion only | 0.2458 | 0.2499 | 0.1666 | 0.1796 | 13.83 |
| ExtraTrees | State only | 0.4882 | 0.2295 | 0.1370 | 0.3372 | 6.93 |
| Random Forest | State only | 0.0902 | 0.2279 | 0.0796 | 0.0335 | 11.11 |

## Interpretation
- Random Forest with combined features achieved the highest macro-F1 (0.1876) and balanced accuracy (0.2689) among these experiments.
- ExtraTrees with combined features had slightly higher raw accuracy (0.2676) and lower training time (12.55 seconds).
- Adding the two state features to motion features improved macro-F1 for both models in this experiment.
- State-only accuracy is not sufficient to judge model quality: its balanced accuracy and macro-F1 were lower than those of the combined models.
- Overall scores remain modest, so reliable recognition across all behavior classes has not been established.

## Limitations
This is an exploratory evaluation on one held-out-driver split, not a validated safety system. Row-level metrics can be affected by differences in recording duration and class distribution across drivers. Target-label provenance and per-driver label concentration require further investigation. Camera-based threshold events and the sensor classifier are separate experimental pipelines, not a fully integrated validated model.

## Conclusion
The combined Random Forest was the strongest model by macro-F1 in this comparison. Future work should investigate class-wise errors, driver-level performance, class imbalance, and validation on additional driver splits before making safety claims.
