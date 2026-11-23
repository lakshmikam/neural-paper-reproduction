# Measured reproduction results

All seeds are reported. Errors are absolute solution errors, not residual loss.

| Case | Runs | Test max error: min / median / max | Median residual SSE |
|---|---:|---:|---:|
| 1 | 3 | 3.367e-07 / 3.664e-06 / 4.910e-06 | 1.227e-11 |
| 2 | 3 | 2.973e-07 / 4.642e-07 / 8.514e-07 | 1.197e-12 |
| 3_ivp | 3 | 6.095e-08 / 1.095e-07 / 1.857e-07 | 4.967e-13 |
| 3_bvp | 3 | 7.675e-10 / 1.795e-09 / 2.257e-09 | 3.535e-13 |
| 4 | 3 | 1.430e-03 / 2.187e-03 / 1.860e+00 | 7.033e-11 |
| 5 | 3 | 5.805e-09 / 6.525e-09 / 8.469e-09 | 2.340e-11 |
| 6 | 3 | 1.608e-03 / 2.249e-03 / 3.747e-03 | 3.666e-04 |
| 7 | 3 | 8.192e-07 / 3.192e-06 / 5.354e-06 | 3.437e-07 |
| 8 | 3 | 1.352e-06 / 6.518e-05 / 1.104e-04 | 6.124e-07 |

## PDE comparison with preprint Table 1

Our NN column uses the median of all seeds on 900 cell-center test points. FEM uses 23×23 endpoint-inclusive points; these are distinct evaluation sets.

| Problem | Paper NN interpolation | Our NN median | Paper FEM nodes / interpolation | Our FEM nodes / interpolation |
|---|---:|---:|---:|---:|
| 5 | 5.000e-07 | 6.525e-09 | 2.000e-08 / 1.500e-05 | 1.834e-08 / 1.165e-05 |
| 6 | 1.500e-03 | 2.249e-03 | 2.000e-04 / 2.500e-03 | 2.032e-04 / 3.298e-03 |
| 7 | 6.000e-06 | 3.192e-06 | 7.000e-07 / 4.000e-05 | 6.370e-07 / 4.165e-05 |
| 8 | 1.500e-05 | 6.518e-05 | 6.000e-07 / 4.000e-05 | 5.231e-07 / 4.166e-05 |

## Optimizer outcomes

18 of 27 runs returned a non-success optimizer status. See per-run JSON for the exact message, iteration count and loss.

Total recorded neural training time: 1568.3 CPU wall-clock seconds.

## Interpretation

Problem 4 exposes a collocation failure: a tiny residual at ten points does not guarantee an accurate solution between them. The seed-0 run converges in optimizer terms but has a large solution error. This result is retained, not filtered out.

BFGS precision-loss warnings can occur even at small solution errors. Accuracy and optimizer termination are reported separately. The difficult oscillatory Problem 6 must be assessed by measured errors, not the architecture alone.

This is an independent numerical reproduction of the 1997 preprint, not an exact recreation of the historical software, figures, or random initialization. See README for protocol differences.
