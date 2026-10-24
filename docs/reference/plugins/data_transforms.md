---
audience: public
generated: reference
---

# Data transforms: reference

Every one of these 1 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `wavelet`

Replace each observed field of `datatype` by its leading wavelet coefficients.

Select it with:

```toml
[[data_transforms]]
name = "wavelet"
```

| option | type | default | description |
|---|---|---|---|
| `datatype` | string | required | |
| `grid` | array | required | |
| `inactive_value` | number or null | | |
| `keep_approximation` | boolean | false | |
| `level` | integer | required | |
| `mask` | ArrayLike or null | | |
| `min_noise` | number | 1e-09 | |
| `noise` | string | white | |
| `order` | string | F | |
| `threshold_multiplier` | number | 1.0 | |
| `threshold_rule` | string | universal | |
| `thresholding` | string | hard | |
| `times` | array or null | | |
| `wavelet` | string | required | |
