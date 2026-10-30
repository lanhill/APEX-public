---
audience: public
items:
  - plugin:variogram_estimators:wls
---

# Variogram estimators

A variogram estimator turns your hard data (measured values at known locations) into a variogram: a summary of how quickly values become unlike each other as the distance between points grows. This page covers the one estimator plugin available in the `variogram_estimators` registry: `wls`.

## Goal

By the end of this page you will know:

- what `wls` computes from your hard data,
- what its two options, `n_lags` and `max_lag`, control,
- how to reason about their values for your own data.

## Prerequisites

- APEX installed as described in the README. Run commands from the APEX folder as `uv run apex ...`.
- Hard data for the variogram to be estimated from. `wls` works on hard data only.
- A basic idea of what a variogram is. No geostatistics expertise is needed beyond that.

## What `wls` does

`wls` estimates a variogram from hard data in two stages:

1. It computes Matheron's semivariogram: the data pairs are grouped into lag classes by the distance between the two points, and each class gets one semivariogram value.
2. It fits a variogram model to those values by weighted least squares.

The lag classes all have the same width and together cover the distances from zero up to `max_lag`.

## Steps

`variogram_estimators` is a plugin registry, not a configuration section. Do not write a `[variogram_estimators]` section. You choose the plugin with `name = "wls"` in the section of the role that uses a variogram estimator, and you give the plugin's options there.

1. In that role's section, set `name = "wls"`.
2. Decide whether the defaults suit your data. Both options are optional.
3. If they do not, set `n_lags` and/or `max_lag` as described below.

### Options

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `n_lags` | integer | `20` | Number of lag classes, of equal width up to `max_lag`. |
| `max_lag` | number or none | none | The largest lag, in cells. When none (the default), it is half the data's horizontal extent. |

No other options exist. `wls` accepts only these two, so any other option name is an error.

#### Why `n_lags` defaults to 20

Twenty is more than the usual dozen. Variogram models differ most at short lags: an exponential model has a steep slope at the origin, and a gaussian model starts flat. A coarse first class hides that difference. In a test on 20 exponential fields, the right model won 9 times with 12 classes and 11 times with 20.

#### What `max_lag` means

`max_lag` is measured in cells, not in metres. Left at its default, it is half of the data's horizontal extent. Pairs of points further apart than this are not used.

## Worked example

Suppose your hard data spans 200 cells horizontally.

- With the defaults, `max_lag` is half the extent: 100 cells. With `n_lags` at 20, each lag class is 100 / 20 = 5 cells wide.
- Suppose you believe the correlation dies out within about 60 cells. Setting `max_lag` to `60` with `n_lags` still at `20` gives classes 60 / 20 = 3 cells wide. The short lags are sampled more finely, which helps the fit tell the models apart near the origin.
- If you only raise `n_lags` to `40` and keep the default `max_lag`, the classes are 100 / 40 = 2.5 cells wide. Each class then holds fewer pairs, so its value is noisier.

The trade-off is the same in every case: narrower classes show more detail at short lags, but each contains fewer data pairs.

## Pitfalls

- **Writing a `[variogram_estimators]` section.** It is a registry of plugins, not a section. Choose the plugin with `name = "wls"` in its role's section instead.
- **Reading `max_lag` as a physical distance.** It is in cells.
- **Setting `max_lag` beyond the data.** The default is half the horizontal extent. A much larger value adds lag classes that few or no pairs can fill.
- **Using too many lag classes on sparse data.** Each class then holds few pairs and its value is noisy.
- **Using too few lag classes.** A coarse first class hides the short-lag behaviour that distinguishes the models. This is why the default is 20, not 12.
- **Expecting `wls` to use soft data.** It estimates variograms from hard data only.

## What to read next

- The other pages in the plugins section of this guide, for the roles that use a variogram estimator.
- The README, for installation and running `uv run apex` from the APEX folder.
