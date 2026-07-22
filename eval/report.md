# Agent evaluation report

_Generated 2026-07-22T08:01:46.094515+00:00_

**13/13 cases passed**

## By category

| category | passed | total |
| --- | --- | --- |
| false_positive | 1 | 1 |
| multi_tool | 1 | 1 |
| rag | 6 | 6 |
| regression | 5 | 5 |

## Forecast benchmark (held-out 20%)

- Test windows: 576
- **LSTM**: MAE 19.24, RMSE 31.089
- **Persistence**: MAE 19.392, RMSE 33.141
- Beats persistence: **True**

## Context relevance (cosine of best expected-source chunk)

| case | cosine |
| --- | --- |
| rag_grap_stage3 | 0.4073 |
| rag_aqi_350 | 0.5425 |
| rag_who_guideline | 0.5409 |
| rag_us_breakpoints | 0.6241 |
| e2e_who_comparison | 0.4518 |

Floor 0.3; min observed 0.4073.

## CI gates

| gate | pass |
| --- | --- |
| regression_pass | ✅ |
| routing_pass | ✅ |
| retrieval_accuracy_pass | ✅ |
| faithfulness_pass | ✅ |
| context_relevance_floor_pass | ✅ |
| context_relevance_drift_pass | ✅ |
| forecast_beats_persistence | ✅ |
