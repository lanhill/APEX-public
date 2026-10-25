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
| `free_surface` | string | stations | `"stations"` (the default): each station is on the half-space's free surface, the seabed offshore (water carries no shear) or the ground onshore, so a cell's depth counts from the station's (`c - z`; the free-surface form). `"datum"`: the free surface is z = 0 and a station at depth z is inside the rock, the reference's (with seabed stations at the water depth: pet_parity.md G13). |
| `poisson` | number | 0.25 | Poisson's ratio nu of the overburden and reservoir, in (-1, 0.5]. |
| `truncation_factor` | number or null | | Zero a station farther from a cell than this times the cell's depth below it, as the reference does with 3; `None` (the default) keeps every station. |

## `van_opstal`

A nucleus of strain over a rigid basement (van Opstal 1974; Tempone et al. 2010).

| option | type | default | description |
|---|---|---|---|
| `basement_depth` | number | required | Depth of the rigid basement (m, positive down, from z = 0), below every cell. |
| `free_surface` | string | stations | As `Geertsma`'s: with `"stations"` each station is on the free surface and the basement's depth counts from the station too. |
| `poisson` | number | 0.25 | Poisson's ratio of the rock, as `Geertsma`'s. |
