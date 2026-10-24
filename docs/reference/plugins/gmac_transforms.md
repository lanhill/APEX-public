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
| `bins` | integer | 100 | |
| `ranges` | object or null | | |
| `variables` | array or null | | |

## `identity`

The paper's I term: the ensemble-whitened state, so that `S_Sᵀ S_S = I_N`.


## `state`

The raw state (or some of its variables): MATLAB `RLM_MAC_L1`'s l₁² on `m`.

| option | type | default | description |
|---|---|---|---|
| `variables` | array or null | | |

## `tv`

First differences along each grid axis, per variable, stacked (`S_V`, `getTV2D`).

| option | type | default | description |
|---|---|---|---|
| `scale_per_variable` | boolean | false | |
| `variables` | array or null | | |
