---
audience: public
generated: reference
---

# Reflectivity: reference

Every one of these 5 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `aki_richards`

Aki & Richards' (1980) linearised PP reflectivity, eq. 5.44.

| option | type | default | description |
|---|---|---|---|
| `form` | string | textbook | `"textbook"` (the default; bruges' `akirichards`):: R = ½(1 - 4Vs_m²p²) Δrho/rho_m + ΔVp/(2Vp_m cos²θ̄) - 4Vs_m²p² ΔVs/Vs_m with `θ̄` the mean of the incidence and transmission angles and `p` the ray parameter. `"linear"` (PyLops' `akirichards` coefficients, linear in the contrasts, as its AVO modelling operator uses):: R = ΔVp/Vp_m / (2 cos²θ₁) - 4 r² sin²θ₁ ΔVs/Vs_m + (½ - 2 r² sin²θ₁) Δrho/rho_m, r = Vs_m/Vp_m Both agree to first order in the contrasts. |

## `approx_zoeppritz_pp`

PyLops' `approx_zoeppritz_pp`, the 3dBox example's choice: the same as `zoeppritz`.


## `fatti`

Fatti et al.'s (1994) form of the linearised reflectivity, in impedance contrasts.


## `shuey`

Shuey's (1985) three-term approximation, `R₀ + G sin²θ + F (tan²θ - sin²θ)`.


## `zoeppritz`

Exact PP reflectivity: Aki & Richards' (1980) explicit solution, eq. 5.40.
