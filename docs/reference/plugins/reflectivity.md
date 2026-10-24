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
| `form` | string | textbook | |

## `approx_zoeppritz_pp`

PyLops' `approx_zoeppritz_pp`, the 3dBox example's choice: the same as `zoeppritz`.


## `fatti`

Fatti et al.'s (1994) form of the linearised reflectivity, in impedance contrasts.


## `shuey`

Shuey's (1985) three-term approximation, `R₀ + G sin²θ + F (tan²θ - sin²θ)`.


## `zoeppritz`

Exact PP reflectivity: Aki & Richards' (1980) explicit solution, eq. 5.40.
