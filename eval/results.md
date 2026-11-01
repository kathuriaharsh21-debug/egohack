# EgoHack Evaluation Report

> [!NOTE]
> Alignment: Matches each labeled frame to the nearest processed frame within +-1 frame.
> Test split opened on **2026-11-01 06:21 IST**; strictly NO pipeline tuning on test results.
> Note on Human Labels: `060-006 f10` is excluded from the human block (flagged as old default pattern).
> Metrics report exact sample sizes ($n=$). Any sample size with $n < 30$ is explicitly flagged as `(too small to trust)`. Metrics with zero labels are marked `N/A`.

> **Spot Check Status**: Merged AI labels not spot-checked.

## Block A: Against Human Labels (Ground Truth, Flagged Defaults Excluded)

### Human Split: `TUNE`
- **Human Labels Evaluated**: 9 (too small to trust)

| Metric | Overall | Easy Frames | Hard Segments |
| :--- | :--- | :--- | :--- |
| **Hand Detection Recall** | 42.9% (3/7, n=7) (too small to trust) | 50.0% (3/6, n=6) (too small to trust) | 0.0% (0/1, n=1) (too small to trust) |
| **False Detection Rate** | 0.0% (0/1, n=1) (too small to trust) | N/A (n=0) | 0.0% (0/1, n=1) (too small to trust) |
| **Side Accuracy (L/R)** | 0.0% (0/3, n=3) (too small to trust) | 0.0% (0/3, n=3) (too small to trust) | N/A (n=0) |
| **Contact State Accuracy** | 0.0% (0/3, n=3) (too small to trust) | 0.0% (0/3, n=3) (too small to trust) | N/A (n=0) |
| **Predicted (visible=false) Hands** | 0.0% (0/9, n=9) (too small to trust) | 0.0% (0/6, n=6) (too small to trust) | 0.0% (0/3, n=3) (too small to trust) |
| **Review Queue Catch Rate** | Queue: 5/7 (71.4%, n=7) (too small to trust) vs Random: 4.1/7 (58.6%) | Queue: 4/6 (66.7%, n=6) (too small to trust) vs Random: 3.2/6 (53.3%) | Queue: 1/1 (100.0%, n=1) (too small to trust) vs Random: 0.9/1 (90.0%) |

#### Contact State Confusion Matrix (Overall)
| GT \ Predicted | none | portable | fixed |
| :--- | :---: | :---: | :---: |
| **none** | 0 | 1 | 0 |
| **portable** | 2 | 0 | 0 |
| **fixed** | 0 | 0 | 0 |

### Human Split: `TEST`
- **Human Labels Evaluated**: 11 (too small to trust)

| Metric | Overall | Easy Frames | Hard Segments |
| :--- | :--- | :--- | :--- |
| **Hand Detection Recall** | 50.0% (3/6, n=6) (too small to trust) | 50.0% (2/4, n=4) (too small to trust) | 50.0% (1/2, n=2) (too small to trust) |
| **False Detection Rate** | 0.0% (0/3, n=3) (too small to trust) | 0.0% (0/1, n=1) (too small to trust) | 0.0% (0/2, n=2) (too small to trust) |
| **Side Accuracy (L/R)** | 75.0% (3/4, n=4) (too small to trust) | 50.0% (1/2, n=2) (too small to trust) | 100.0% (2/2, n=2) (too small to trust) |
| **Contact State Accuracy** | 25.0% (1/4, n=4) (too small to trust) | 0.0% (0/2, n=2) (too small to trust) | 50.0% (1/2, n=2) (too small to trust) |
| **Predicted (visible=false) Hands** | 0.0% (0/11, n=11) (too small to trust) | 0.0% (0/5, n=5) (too small to trust) | 0.0% (0/6, n=6) (too small to trust) |
| **Review Queue Catch Rate** | Queue: 5/6 (83.3%, n=6) (too small to trust) vs Random: 4.8/6 (80.0%) | Queue: 3/4 (75.0%, n=4) (too small to trust) vs Random: 3.0/4 (75.0%) | Queue: 2/2 (100.0%, n=2) (too small to trust) vs Random: 1.9/2 (95.0%) |

#### Contact State Confusion Matrix (Overall)
| GT \ Predicted | none | portable | fixed |
| :--- | :---: | :---: | :---: |
| **none** | 1 | 0 | 0 |
| **portable** | 2 | 0 | 0 |
| **fixed** | 1 | 0 | 0 |

---
## Block B: Against Merged AI Labels (from earlier AI passes, not an independent look)

> [!WARNING]
> Merged AI labels are aggregated from earlier AI passes with sides omitted. They provide secondary pipeline stress-testing but are NOT human ground truth.

### Merged-AI Split: `TUNE`
- **Merged-AI Labels Evaluated**: 30

| Metric | Overall | Easy Frames | Hard Segments |
| :--- | :--- | :--- | :--- |
| **Hand Detection Recall** | 92.6% (25/27, n=27) (too small to trust) | 91.7% (22/24, n=24) (too small to trust) | 100.0% (3/3, n=3) (too small to trust) |
| **False Detection Rate** | 100.0% (1/1, n=1) (too small to trust) | N/A (n=0) | 100.0% (1/1, n=1) (too small to trust) |
| **Side Accuracy (L/R)** | N/A (omitted on merged AI labels) | N/A | N/A |
| **Contact State Accuracy** | 12.5% (3/24, n=24) (too small to trust) | 10.0% (2/20, n=20) (too small to trust) | 25.0% (1/4, n=4) (too small to trust) |
| **Predicted (visible=false) Hands** | 0.0% (0/30, n=30) | 0.0% (0/26, n=26) (too small to trust) | 0.0% (0/4, n=4) (too small to trust) |
| **Review Queue Catch Rate** | Queue: 16/24 (66.7%, n=24) (too small to trust) vs Random: 17.2/24 (71.7%) | Queue: 14/20 (70.0%, n=20) (too small to trust) vs Random: 14.2/20 (71.0%) | Queue: 2/4 (50.0%, n=4) (too small to trust) vs Random: 3.2/4 (80.0%) |

#### Contact State Confusion Matrix (Overall)
| AI-Label \ Predicted | none | portable | fixed |
| :--- | :---: | :---: | :---: |
| **none** | 2 | 0 | 0 |
| **portable** | 21 | 1 | 0 |
| **fixed** | 0 | 0 | 0 |

### Merged-AI Split: `TEST`
- **Merged-AI Labels Evaluated**: 49

| Metric | Overall | Easy Frames | Hard Segments |
| :--- | :--- | :--- | :--- |
| **Hand Detection Recall** | 93.0% (40/43, n=43) | 92.7% (38/41, n=41) | 100.0% (2/2, n=2) (too small to trust) |
| **False Detection Rate** | 0.0% (0/1, n=1) (too small to trust) | 0.0% (0/1, n=1) (too small to trust) | N/A (n=0) |
| **Side Accuracy (L/R)** | N/A (omitted on merged AI labels) | N/A | N/A |
| **Contact State Accuracy** | 8.3% (3/36, n=36) | 8.8% (3/34, n=34) | 0.0% (0/2, n=2) (too small to trust) |
| **Predicted (visible=false) Hands** | 0.0% (0/49, n=49) | 0.0% (0/46, n=46) | 0.0% (0/3, n=3) (too small to trust) |
| **Review Queue Catch Rate** | Queue: 24/36 (66.7%, n=36) vs Random: 24.0/36 (66.7%) | Queue: 23/34 (67.6%, n=34) vs Random: 22.1/34 (65.0%) | Queue: 1/2 (50.0%, n=2) (too small to trust) vs Random: 1.6/2 (80.0%) |

#### Contact State Confusion Matrix (Overall)
| AI-Label \ Predicted | none | portable | fixed |
| :--- | :---: | :---: | :---: |
| **none** | 0 | 0 | 0 |
| **portable** | 32 | 3 | 0 |
| **fixed** | 1 | 0 | 0 |
