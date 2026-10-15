
### MRI (MSD Task01, 98 held-out patients; mean over seeds)

| Method | Dice WT (avg 15) | Dice TC (avg 15) | Dice ET (avg 15) | Dice mean (all 4 seq.) | node ECE ↓ | AUROC err | u boundary / interior |
|---|---|---|---|---|---|---|---|
| MUSE | **0.587** | 0.320 | 0.258 | 0.538 | 0.120 | 0.666 | 1.18 |
| GAT-zero | 0.536 | **0.327** | **0.271** | **0.590** | **0.081** | **0.786** | 1.38 |

#### Mean Dice per available-sequence pattern

| Pattern | MUSE | GAT-zero |
|---|---|---|
| FLAIR | **0.325** | 0.215 |
| T1 | **0.173** | 0.126 |
| T1gd | **0.357** | 0.351 |
| T2 | **0.231** | 0.206 |
| FLAIR+T1 | **0.332** | 0.276 |
| FLAIR+T1gd | **0.526** | 0.515 |
| FLAIR+T2 | **0.325** | 0.295 |
| T1+T1gd | **0.409** | 0.378 |
| T1+T2 | 0.260 | **0.274** |
| T1gd+T2 | 0.463 | **0.474** |
| FLAIR+T1+T1gd | **0.552** | 0.545 |
| FLAIR+T1+T2 | 0.321 | **0.331** |
| FLAIR+T1gd+T2 | 0.537 | **0.583** |
| T1+T1gd+T2 | 0.474 | **0.506** |
| FLAIR+T1+T1gd+T2 | 0.538 | **0.590** |