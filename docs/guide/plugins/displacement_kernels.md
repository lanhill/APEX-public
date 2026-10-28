---
audience: public
items:
  - cli:apex run
  - plugin:displacement_kernels:geertsma
  - plugin:displacement_kernels:van_opstal
---

# Displacement kernels

## Goal

A displacement kernel is the elastic response used to turn a source of strain into displacement. APEX provides two in the `displacement_kernels` registry:

- `geertsma`: Geertsma's (1973) nucleus of strain in an elastic half-space.
- `van_opstal`: a nucleus of strain over a rigid basement (van Opstal 1974; Tempone et al. 2010).

By the end of this page you will be able to choose between them, set each one in a configuration, and check that the configuration is accepted.

## Prerequisites

- A history-matching configuration with seafloor displacement data: a `[[geophysics.data]]` entry with `kind = "seafloor_displacement"`.
- A decision about whether your model is a plain half-space (`geertsma`) or has a rigid basement at a known depth (`van_opstal`).
- For `van_opstal`, the depth of the rigid basement in metres. It is required. A rigid basement is a layer at depth that is treated as not deforming. The depth is measured positive down from z = 0 and must lie below every cell of your model.

## Steps

1. **Pick the kernel.** Use `geertsma` for an elastic half-space. Use `van_opstal` when a rigid basement lies below every cell.
2. **Choose it in the data's options.** The kernel is the `kernel` option of the entry's `[geophysics.data.options]`: `kernel = { name = "geertsma", options = { ... } }`, or `kernel = "geertsma"` to keep a kernel's defaults. Without `kernel`, the entry uses `geertsma` with its defaults.
3. **Set the options.** Only the options below exist. Any other option is rejected, because neither plugin accepts additional properties.
4. **Leave out what you don't need.** Every option except `basement_depth` has a default.

### Options for `geertsma`

| Option | Type | Default | Notes |
|---|---|---|---|
| `poisson` | number | `0.25` | Poisson's ratio of the overburden and reservoir, in (-1, 0.5]. |
| `truncation_factor` | number or unset | unset | When unset, every station is kept (no truncation). See below. |
| `free_surface` | `"stations"` or `"datum"` | `"stations"` | Where the free surface is. See below. |

### Options for `van_opstal`

| Option | Type | Default | Notes |
|---|---|---|---|
| `basement_depth` | number | none (required) | Depth of the rigid basement in metres, positive down from z = 0, below every cell. You must set it. |
| `poisson` | number | `0.25` | Poisson's ratio of the rock, as for `geertsma`. |
| `free_surface` | `"stations"` or `"datum"` | `"stations"` | As for `geertsma`. With `"stations"`, the basement's depth counts from the station too. |

Note that `truncation_factor` belongs to `geertsma` only. `van_opstal` does not accept it.

### What `truncation_factor` does

By default `truncation_factor` is unset, and then every station contributes for every cell. If you give it a number, a station farther from a cell than that number times the cell's depth below it is set to zero for that cell. The reference implementation uses 3. Set a number when you want to reproduce that behaviour or to ignore very distant stations. Otherwise leave it out.

### What `free_surface` means

- `"stations"` (the default): each station sits on the free surface of the half-space, that is the seabed offshore (water carries no shear) or the ground onshore. A cell's depth is counted from the station's depth (`c - z`).
- `"datum"`: the free surface is the plane z = 0, and a station at depth z is inside the rock. This is the reference implementation's choice, with seabed stations at the water depth.

Pick `"stations"` for stations on the seabed or ground. Pick `"datum"` when you want to match a reference that puts the free surface at z = 0.

## Worked example

First, a half-space kernel with a different Poisson's ratio and the free surface at the datum, to match a reference that puts it at z = 0. `truncation_factor` is left out, so it stays unset and no station is dropped.

```toml
[[geophysics.data]]
kind = "seafloor_displacement"
datatype = "subsidence"
vintages = ["2025-01-01"]

[geophysics.data.options]
baseline = "2020-01-01"
stations = { spacing = 500.0, depth = 300.0 }
compressibility = 1e-10
kernel = { name = "geertsma", options = { poisson = 0.3, free_surface = "datum" } }
```

Next, a rigid-basement kernel with the basement 3000 m deep. Replace 3000.0 with the depth for your own model. Only the required `basement_depth` is set, so `poisson` stays `0.25` and `free_surface` stays `"stations"`.

```toml
[[geophysics.data]]
kind = "seafloor_displacement"
datatype = "subsidence"
vintages = ["2025-01-01"]

[geophysics.data.options]
baseline = "2020-01-01"
stations = { spacing = 500.0, depth = 300.0 }
compressibility = 1e-10
kernel = { name = "van_opstal", options = { basement_depth = 3000.0 } }
```

A half-space kernel with every default needs only its name: `kernel = "geertsma"` (or leave `kernel` out).

## Pitfalls

- **Missing `basement_depth`.** `van_opstal` has no default for it, so `kernel = "van_opstal"` alone is refused.
- **Basement not below the cells.** `basement_depth` must be below every cell, measured from z = 0.
- **Wrong option for the kernel.** `truncation_factor` is valid for `geertsma` only, and `basement_depth` for `van_opstal` only. Unknown options are rejected.
- **Misspelled `free_surface`.** Only `"stations"` and `"datum"` are accepted, in lowercase and quoted.
- **Assuming the two kernels give the same answer.** They model different media, so switching from `geertsma` to `van_opstal` changes the physics, not just the settings.
- **Forgetting the defaults.** If you do not set `poisson`, it is `0.25`. Set it explicitly if your medium differs.

## What to read next

To confirm that your configuration is accepted, start a short run that stops after one analysis step (replace `config.toml` with your file):

```bash
uv run apex run config.toml --max-steps 1
```

A configuration with a rejected option or a missing `basement_depth` is refused. Add `--no-progress` if you do not want progress on standard error. To continue a stopped run, use `apex resume`.

Then continue with the other plugin pages in the guide, which show how the geophysical data fit into a complete configuration.
