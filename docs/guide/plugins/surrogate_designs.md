---
audience: public
items:
  - plugin:surrogate_designs:file
  - plugin:surrogate_designs:lhs
  - plugin:surrogate_designs:prior
  - plugin:surrogate_designs:run
---

# Surrogate designs

A surrogate design is the plan that decides which input points a surrogate is built from. APEX's core provides four design plans in the `surrogate_designs` registry: `prior`, `lhs`, `file` and `run`. This page explains how they relate, so you know which name to pick and what to expect from it.

## Goal

By the end of this page you will know:

- which design plans core provides,
- that `lhs`, `file` and `run` are aliases of `prior`,
- that none of these plans has options of its own.

## Prerequisites

- APEX installed as described in the README. Commands are run from the APEX folder as `uv run apex ...`.
- A surrogate workflow in which you choose a design plan by name.

## The four plans

| Plan | What it is |
|------|------------|
| `prior` | Core's own design plan. |
| `lhs` | An alias of `prior`. |
| `file` | An alias of `prior`. |
| `run` | An alias of `prior`. |

An alias is another name for the same plan. Choosing `lhs`, `file` or `run` gives you the same design plan as choosing `prior`. The names exist so that a configuration can say what you intend (a Latin hypercube, a file of points, a run), while core resolves all of them to `prior`.

## Options

Each of the four plans accepts no options. The option set for `prior`, `lhs`, `file` and `run` is empty, and unknown options are rejected. Do not add option keys under any of these plans; there is nothing to tune on them.

## Steps

1. Decide which name best describes your intent: `prior`, `lhs`, `file` or `run`.
2. Use that name wherever your surrogate workflow asks for a design plan.
3. Leave the plan's options empty.

## Worked example

Suppose you want a configuration that reads naturally as a space-filling design. You name the plan `lhs`. Because `lhs` is an alias of `prior`, the result is identical to naming `prior`. If a colleague later changes the name to `prior`, nothing about the design changes. Likewise, naming `file` or `run` does not by itself make APEX read points from a file or from a run: all four names resolve to the same core plan.

## Pitfalls

- **Expecting different behaviour from different names.** `lhs`, `file` and `run` are aliases of `prior`. They do not behave differently from it.
- **Adding options.** The plans have no options, and extra keys are not accepted.
- **Assuming the name configures a sampling method or an input source.** The name selects the plan; it carries no extra settings.

## What to read next

- The other pages in the plugins section of the guide, to see which surrogate plugins consume a design plan.
- The guide's configuration pages, for where plugin choices are written in your configuration.
