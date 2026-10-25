---
audience: public
generated: reference
---

# Variogram estimators: reference

This item, with its options, as APEX's code defines it. Written from the code; the user guide teaches their use.

## `wls`

Variograms from hard data: Matheron's semivariogram, fitted by weighted least squares.

| option | type | default | description |
|---|---|---|---|
| `max_lag` | number or null | | The largest lag, in cells; none (the default): half the data's horizontal extent. |
| `n_lags` | integer | 20 | Lag classes, of equal width up to `max_lag`. Twenty, not the usual dozen: the models differ most at short lags (an exponential's slope at the origin, a gaussian's flat start), which a coarse first class hides (on 20 exponential fields the right model won 9 times with 12 classes, 11 with 20). |
