---
audience: public
items:
  - cli:apex plugins
  - cli:apex report
  - cli:apex resume
  - cli:apex run
  - cli:apex stop
  - plugin:avo_attributes:max
  - plugin:avo_attributes:mean
  - plugin:avo_attributes:min
  - plugin:avo_attributes:rms
  - plugin:avo_attributes:sum_negative
  - plugin:avo_attributes:time_shift
  - plugin:avo_attributes:trace
---

# Avo attributes

The `avo_attributes` registry holds seven plugins: `max`, `mean`, `min`, `rms`, `sum_negative`, `time_shift` and `trace`. Most of them reduce the samples of a trace inside a window to one amplitude attribute. `trace` returns the windowed trace itself, and `time_shift` returns a two-way time instead of an amplitude. This page explains what each one computes and how to choose between them.

## Goal

By the end of this lesson you will be able to:

- say what each of the seven plugins computes over a window;
- pick the right one for a question such as "where is the strongest peak?" or "is there a soft-sand or gas indication?";
- work out by hand what each amplitude attribute returns for a few samples, and what `trace` returns for a longer trace;
- know when `time_shift` is the right choice, and what kind of value it gives you.

## Prerequisites

- A basic idea of amplitude versus offset (AVO) analysis, and of a seismic trace as a series of amplitude samples.
- An idea of the window: the contiguous range of samples an attribute looks at, for example "the 3rd to the 7th sample of the trace". The attribute works only on the samples inside it. None of these plugins has an option for the window, so you cannot set the window on the plugin itself. The examples on this page simply assume a window is given.
- Access to the `apex` command line, to check which plugins are installed and to start a run.

## The plugins

None of these plugins has any options. Each one's option set is empty and accepts nothing extra, so there is nothing to tune. You choose the behaviour by choosing the plugin.

| Plugin | What it returns over the window |
|---|---|
| `max` | The most positive amplitude (a peak). |
| `min` | The most negative amplitude (a trough). |
| `mean` | The mean amplitude. |
| `rms` | The root mean square amplitude, `sqrt(mean(s²))`. This is the default. |
| `sum_negative` | The sum of the negative amplitudes (a soft-sand or gas indicator). |
| `time_shift` | The two-way time to the reservoir's base, for the time shift. |
| `trace` | The whole trace: the samples in the window, and zero outside it. |

## Steps

1. Decide the question you are asking of the window.
2. Pick the plugin:
   - Strongest positive reflection: `max`.
   - Strongest negative reflection: `min`.
   - Overall bias of the window: `mean`.
   - Overall energy, regardless of sign: `rms`. It is the default, so it is what you get if you do not choose another.
   - Evidence of soft sand or gas: `sum_negative`.
   - The two-way time to the reservoir's base, for the time shift: `time_shift`.
   - The samples themselves, not a single number: `trace`.
3. Check that the plugin is installed. Run:

   ```
   uv run apex plugins
   ```

   This lists every plugin with its options. Look for the `avo_attributes` plugins and the name you picked. For these plugins the options are empty. If the name you picked is not listed, it is not available in your installation, so pick one that is. For the full descriptions as JSON, add the `--json` flag:

   ```
   uv run apex plugins --json
   ```

4. Put the plugin name (for example `max`) where your run configuration names its AVO attribute, and give it no options. The run configuration is the TOML file you pass to `apex run`. The facts this page is built on do not say which section or key of that file holds the attribute, so this page cannot show a complete configuration. The safe route is to start from a history-matching configuration that already runs for you and change only the attribute's plugin name, leaving everything else as it is.
5. Start the run:

   ```
   uv run apex run config.toml
   ```

   To test on a small scale, stop early with `--max-steps`. The run stops after that many analysis steps, and you can continue later with `apex resume` and the run directory:

   ```
   uv run apex run config.toml --max-steps 1
   ```

   Progress is shown on standard error. Use `--no-progress` to turn it off (`--progress` is the default).
6. Look at the result of the run. `apex report` prints the data mismatch history and the performance metrics of a run. Give it the run directory:

   ```
   uv run apex report RUN_DIR
   ```

   Add `--plots` to also write the result figures as PNG and PDF. They go to `RUN_DIR/report` unless you give another directory with `--output`:

   ```
   uv run apex report RUN_DIR --plots --output figures
   ```

   This shows how the mismatch behaves when you change plugin. To check what an attribute computes, use the hand calculations below, which need no run.

## Worked example

Suppose the window contains five samples:

```
0.2  -0.5  0.4  -0.1  0.6
```

The amplitude attributes are:

- `max`: the largest value is **0.6**.
- `min`: the smallest value is **-0.5**.
- `mean`: (0.2 − 0.5 + 0.4 − 0.1 + 0.6) / 5 = 0.6 / 5 = **0.12**.
- `rms`: the squares are 0.04, 0.25, 0.16, 0.01 and 0.36, which sum to 0.82. The mean is 0.164, and its square root is about **0.405**.
- `sum_negative`: the negative samples are -0.5 and -0.1, so the sum is **-0.6**.

Notice that `mean` is small (0.12) even though the window contains large swings. `rms` (about 0.405) reflects the size of those swings, because squaring removes the sign.

### The `trace` plugin on a longer trace

`trace` keeps the samples in the window and sets the samples outside it to zero. Take a trace of nine samples and a window that covers the 3rd to the 7th sample:

```
Input trace:   0.1  -0.3   0.2  -0.5   0.4  -0.1   0.6   0.3  -0.2
Window:                    [ 0.2  -0.5   0.4  -0.1   0.6 ]
Output (trace): 0    0     0.2  -0.5   0.4  -0.1   0.6    0     0
```

The five samples inside the window are kept unchanged. The four outside it (the first two and the last two) are set to **0**.

### The `time_shift` plugin

`time_shift` does not average or sum amplitudes. It returns one time value: the two-way time to the reservoir's base, which is used for the time shift. Choose `time_shift` when you want that time rather than an amplitude from the window. It has no options. This page does not describe how the time shift is applied or where the base time comes from.

As an illustration only, with made-up numbers: if the reservoir's base lies at a two-way time of 2.150 s, `time_shift` returns 2.150 s. Because the result is a time and not a function of the amplitude statistics above, there is no sample arithmetic to check by hand. Treat it as a time, and do not compare it with the other attributes.

## Pitfalls

- **Expecting options.** These plugins have none. Their option sets accept no properties, so there is no threshold or parameter to adjust. A setting you add anyway is not valid for the plugin. Remove it and choose a different plugin to get different behaviour.
- **Looking for a window setting on the plugin.** The window is not an option of any of these plugins.
- **Using `mean` to judge strength.** Positive and negative samples cancel in the mean. Use `rms` for overall energy.
- **Confusing `min` with `sum_negative`.** `min` is one value, the deepest trough. `sum_negative` adds up every negative sample, so it grows with the number of negative samples as well as their size.
- **Forgetting that `rms` is the default.** If you want a peak or trough, choose `max` or `min` explicitly.
- **Reading `trace` as a single number.** It returns samples. Inside the window they are unchanged, and outside it they are zero.
- **Mixing up `time_shift` with an amplitude attribute.** It returns a two-way time, so do not average it with, or plot it on the same scale as, amplitudes.
- **Treating `sum_negative` as proof.** It is an indicator of soft sand or gas, not a confirmation. Compare it with other evidence.

## What to read next

- Run `uv run apex plugins` to see every plugin and its options, including those in other registries, and `uv run apex plugins --json` for the full descriptions in a form other tools can read.
- Run `uv run apex run config.toml --max-steps 1` on a small case, then continue with `uv run apex resume RUN_DIR`. If you need to pause a run that is going, `uv run apex stop RUN_DIR` asks it to stop at its next checkpoint, and `apex resume` continues it.
- Inspect a finished or partial run with `uv run apex report RUN_DIR`.
- Read the other plugin pages in this guide's plugins section, to see which processing steps you can combine with these attributes.
