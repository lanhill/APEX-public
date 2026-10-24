---
audience: public
generated: reference
---

# Stopping rules: reference

Every one of these 2 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `patience`

Stop after `patience` accepted iterates without a new CV minimum.

Select it with:

```toml
[validation.stopping]
name = "patience"
```

| option | type | default | description |
|---|---|---|---|
| `patience` | integer | 2 | |

## `select`

Never stop early; keep the minimum-CV iterate of the whole path (the paper's protocol).

Select it with:

```toml
[validation.stopping]
name = "select"
```
