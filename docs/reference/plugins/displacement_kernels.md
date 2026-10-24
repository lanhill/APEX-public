---
audience: public
generated: reference
---

# Displacement kernels: reference

Every one of these 2 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `geertsma`

Geertsma's (1973) nucleus of strain in an elastic half-space.

| option | type | default | description |
|---|---|---|---|
| `free_surface` | string | stations | |
| `poisson` | number | 0.25 | |
| `truncation_factor` | number or null | | |

## `van_opstal`

A nucleus of strain over a rigid basement (van Opstal 1974; Tempone et al. 2010).

| option | type | default | description |
|---|---|---|---|
| `basement_depth` | number | required | |
| `free_surface` | string | stations | |
| `poisson` | number | 0.25 | |
