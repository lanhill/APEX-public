---
audience: public
items:
  - plugin:forecast_scores:coverage_p10_p90
  - plugin:forecast_scores:crps
  - plugin:forecast_scores:mean_error
  - plugin:forecast_scores:spread
---

# Forecast scores

This page covers 4 of APEX's functionalities: what each is, how to select it, and its options.

## `coverage_p10_p90`

Fraction of times at which the truth lies in the ensemble's [p10, p90] band.


## `crps`

Continuous ranked probability score, averaged over time (lower is better).


## `mean_error`

RMSE over time of the ensemble mean against the truth.


## `spread`

Ensemble standard deviation, averaged over time.
