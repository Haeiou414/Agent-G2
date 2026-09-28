# Exact tokenizer prefix-depth audit

Model tokenizer: `xiamoent/Agent-G2-alfworld-1.5b` at revision `4556a9bfdf84320267c3a9e9e7b85732ba2835ba`  
Tokenizer SHA-256: `9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa`  
Dataset SHA-256: `1f9d2bd6b9a5a88c06a99f521c63828c00a5494ed346aa2953fcc5764f749a66`

This audit reproduces the released runtime's former token-based prefix calculation with the exact tokenizer, then compares it with the paper's action-count equation.

| Ratio | Different trajectories | Mismatch | Token prefix shallower | Token prefix deeper | Mean absolute step difference |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 309 | 8.7% | 0 | 309 | 0.087 |
| 0.2 | 488 | 13.7% | 5 | 483 | 0.137 |
| 0.3 | 475 | 13.4% | 281 | 194 | 0.134 |
| 0.4 | 503 | 14.2% | 16 | 487 | 0.142 |
| 0.5 | 1,278 | 36.0% | 0 | 1,278 | 0.362 |
| 0.6 | 754 | 21.2% | 266 | 488 | 0.212 |
| 0.7 | 350 | 9.9% | 56 | 294 | 0.099 |
| 0.8 | 170 | 4.8% | 1 | 169 | 0.048 |
| 0.9 | 0 | 0.0% | 0 | 0 | 0.000 |

At ratio 0.5, the two implementations choose different action prefixes for **1,278/3,553 trajectories (36.0%)**.

Generated with `python -m reproduction.analyze_tokenizer_prefix`. The model weights are not downloaded; only the tokenizer artifact is required.
