---
audience: public
items:
  - plugin:geomodeling_skills:continuous
  - plugin:geomodeling_skills:correlated_continuous
  - plugin:geomodeling_skills:facies_mps
  - plugin:geomodeling_skills:facies_pluri
---

# Geomodeling skills

## Goal

This lesson helps you choose the right geomodeling skill for the property you want to model. APEX groups these skills in a plugin registry called `geomodeling_skills`. It holds four plugins:

| Plugin | What it produces |
|---|---|
| `continuous` | Gaussian fields of continuous properties (ln permeability, porosity, ...). |
| `correlated_continuous` | Correlated Gaussian fields (e.g. porosity and ln permeability), as a stored ensemble. |
| `facies_mps` | Facies from a training image (MPSlib's SNESIM), as signed distances, with values. |
| `facies_pluri` | Facies by proportion from a truncated latent Gaussian field, conditioned to wells. |

By the end you will be able to say which of the four fits your problem.

## Prerequisites

- APEX installed as described in the README. Run commands from the APEX folder, starting with `uv run apex ...`.
- A clear idea of what you want to model: a continuous property, several correlated properties, or facies.
- For `facies_mps`, a training image. For `facies_pluri`, well data to condition to.

## Steps

1. **Decide whether the property is continuous or categorical.**
   - Continuous, such as ln permeability or porosity: use `continuous` or `correlated_continuous`.
   - Categorical, meaning facies: use `facies_mps` or `facies_pluri`.
2. **For continuous properties, decide whether the properties must be correlated.**
   - One property on its own: `continuous` gives you Gaussian fields.
   - Properties that must stay correlated with each other, for example porosity and ln permeability: `correlated_continuous`. It produces the fields as a stored ensemble.
3. **For facies, decide where the spatial pattern comes from.**
   - From a training image: `facies_mps`. It uses MPSlib's SNESIM and gives facies as signed distances, with values.
   - From facies proportions: `facies_pluri`. It truncates a latent Gaussian field and conditions the result to wells.
4. **Check the options.** None of the four plugins has options of its own. Each one's option set is empty and accepts no additional properties. There is nothing to tune on the plugin itself, and you should not add keys for it.

## Worked example

Suppose you need porosity and ln permeability for a reservoir model, and the two must be consistent with each other.

- Both are continuous, so you are in the Gaussian-field family (step 1).
- They must be correlated, so `correlated_continuous` is the right choice rather than running `continuous` twice (step 2). It returns the correlated fields as a stored ensemble.

Now suppose you also need facies, and you have well data with known facies proportions but no training image. `facies_pluri` fits: it draws facies by proportion from a truncated latent Gaussian field and conditions them to the wells (step 3). If you had a training image instead, `facies_mps` would be the match.

## Pitfalls

- **Using `continuous` for properties that must be correlated.** Independent fields will not honour the relationship between, say, porosity and ln permeability. Use `correlated_continuous`.
- **Mixing up the two facies plugins.** `facies_mps` is driven by a training image. `facies_pluri` is driven by proportions and wells. They answer different questions.
- **Looking for options.** The plugins expose none, so there is nothing to set on them directly.

## What to read next

- The README, for installation and for running `uv run apex ...` from the APEX folder.
- The other pages in the guide's plugins section, to see how the rest of APEX's plugin registries are organised.
