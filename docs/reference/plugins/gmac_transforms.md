---
audience: public
generated: reference
---

# Gmac transforms: reference

Every one of these 4 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `histogram_entropy`

A histogram's count-weighted entropy per bin, per variable (`S_H`, `getHistEntropy`).

| option | type | default | description |
|---|---|---|---|
| `bins` | integer | 100 | Bins per variable (MATLAB: 100). |
| `ranges` | object or null | | Fixed `[low, high]` per variable (5Spot MATLAB: PORO [0.1, 0.4], PERMX [6, 8]); default the ensemble's and the mean's range at each update. The bins are the same for every member and the mean, so the differences compare like with like. Values outside a fixed range fall in its end bins. |
| `variables` | array or null | | The state variables; default all. |

## `identity`

The paper's I term: the ensemble-whitened state, so that `S_Sᵀ S_S = I_N`.


## `state`

The raw state (or some of its variables): MATLAB `RLM_MAC_L1`'s l₁² on `m`.

| option | type | default | description |
|---|---|---|---|
| `variables` | array or null | | The state variables; default all. |

## `tv`

First differences along each grid axis, per variable, stacked (`S_V`, `getTV2D`).

| option | type | default | description |
|---|---|---|---|
| `scale_per_variable` | boolean | false | Divide each variable's differences by their ensemble RMS, so that no variable dominates by its units (the 5Spot MATLAB stacked PORO and PERMX unscaled: off). |
| `variables` | array or null | | The gridded state variables; default every one with a grid. |
