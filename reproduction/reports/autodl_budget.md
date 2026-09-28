# AutoDL sequential rental estimate

Source: completed 8-step smoke run.
Estimated one-time overhead: 1.7 min/run.
Estimated base step: 109.3 s.
Estimated validation event: 41.4 s.
Estimated checkpoint event: 9.9 s.

| Plan | Runs | Sequential wall-hours | GPU-hours | Estimated fee |
|---|---:|---:|---:|---:|
| 主方法闭环 | 9 | 22.98 | 22.98 | ¥43.21 |
| 完整实验矩阵 | 18 | 45.96 | 45.96 | ¥86.41 |

Per-run target: 80 steps, 8 validation events, 2 checkpoint events.
This is a planning estimate from one measured host, not a promised runtime. Keep a 20–30% balance reserve for downloads, compilation, retries, and host variance.
