---
audience: public
generated: reference
---

# Data transforms: reference

This item, with its options, as APEX's code defines it. Written from the code; the user guide teaches their use.

## `wavelet`

Replace each observed field of `datatype` by its leading wavelet coefficients.

Select it with:

```toml
[[data_transforms]]
name = "wavelet"
```

| option | type | default | description |
|---|---|---|---|
| `datatype` | string | required | Data type of the fields to compress. |
| `grid` | array | required | Shape of the grid the fields live on, e.g. `[nx, ny, nz]`: 1 to 3 positive sizes. |
| `inactive_value` | number or null | | Value for inactive cells; default each observed field's mean (as PET). |
| `keep_approximation` | boolean | false | Keep every approximation coefficient, unthresholded (PET's `keep_ca`); off by default. |
| `level` | integer | required | Number of decomposition levels, at least 1. |
| `mask` | ArrayLike or null | | Active-cell mask (grid-shaped or flat in `order`) shared by the fields; default all cells. Configurations give `mask_file`, which the composition root loads. |
| `min_noise` | number | 1e-09 | Floor on the estimated noise standard deviations, positive (default 1e-9). |
| `noise` | string | white | `"white"` (default): one noise level, estimated from the finest diagonal sub-band, for every coefficient. `"coloured"`: one noise level per sub-band. |
| `order` | string | F | Cell order of the flattened fields and mask: `"F"` (default; the Eclipse order, i fastest) or `"C"`. |
| `threshold_multiplier` | number | 1.0 | Scales every threshold (PET's `th_mult`), at least 0; the default 1 uses the rule's threshold as is, and 0 keeps every non-zero coefficient. |
| `threshold_rule` | string | universal | `"universal"` (default; `sqrt(2 ln n)` times the noise level) or `"bayesian"`, which needs `noise = "coloured"`. Non-zero coefficients at least as large as the threshold in magnitude are retained. |
| `thresholding` | string | hard | `"hard"` (default) keeps the retained coefficients as they are; `"soft"` shrinks them towards zero by the threshold. |
| `times` | array or null | | Only compress the fields at these times; default all. |
| `wavelet` | string | required | A discrete PyWavelets wavelet name, e.g. `"haar"` or `"db2"`. |
