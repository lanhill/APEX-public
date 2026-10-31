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

The `avo_attributes` registry holds seven plugins: `max`, `mean`, `min`, `rms`, `sum_negative`, `time_shift` and `trace`. Most of them reduce the samples of a trace inside a window to one amplitude attribute. `trace` returns the windowed trace itself, and `time_shift` returns a two-way time instead of an amplitude. This lesson explains what each one computes, how to choose between them, and how to compare two of them in a run.

## Goal

By the end of this lesson you will be able to:

- say what each of the seven plugins computes over a window;
- pick the right one for a question such as "where is the strongest peak?" or "is there a soft-sand or gas indication?";
- work out by hand what each amplitude attribute returns for a few samples, and what `trace` returns for a longer trace;
- tell when `time_shift` is the right choice and what kind of value it gives you;
- compare two attributes by running the same case twice and reading both reports.

## Prerequisites

- A basic idea of amplitude versus offset (AVO) analysis, and of a seismic trace as a series of amplitude samples.
- An idea of the window: the contiguous range of samples an attribute looks at, for example "the 3rd to the 7th sample of the trace". The attribute works only on the samples inside it.
- Access to the `apex` command line from the APEX folder, to check which plugins are installed and to start a run.
- A working run configuration (the TOML file you pass to `apex run`).

None of these plugins has an option for the window, so you never set it on the plugin. The window belongs to the run setup, not to the attribute. If you start from a configuration that already runs and change only which attribute plugin it names, the window and everything else stay as they were. The worked examples below simply assume a window is given.

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

   This lists every plugin with its options. Find the `avo_attributes` group and the name you picked. Each of these plugins is listed with no options. If the name you picked is not listed, it is not available in your installation, so pick one that is. For the full descriptions as JSON, add the `--json` flag:

   ```
   uv run apex plugins --json
   ```

4. In your run configuration, name the plugin you picked (for example `max`) where the configuration selects the AVO attribute, and give it no other options. Start from a configuration that already runs, and change only that name.
5. Start the run:

   ```
   uv run apex run config.toml
   ```

   To test on a small scale, stop early with `--max-steps`. The run stops after that many analysis steps (one step is one round of the run's analysis), and you can continue later with `apex resume` and the directory of the run:

   ```
   uv run apex run config.toml --max-steps 1
   ```

   Progress is shown on standard error. Use `--no-progress` to turn it off (`--progress` is the default).
6. Look at the result of the run. `apex report` prints the data mismatch history and the performance metrics of a run. The data mismatch measures how far the data computed from the current model is from the data you matched against, and the history shows how it changes from one analysis step to the next. Give the command the directory of the run:

   ```
   uv run apex report RUN_DIR
   ```

   Add `--plots` to also write the result figures as PNG and PDF. They go to `RUN_DIR/report` unless you give another directory with `--output`:

   ```
   uv run apex report RUN_DIR --plots --output figures
   ```

7. To compare two plugins, make a copy of the configuration for each, change only the attribute plugin in the copies, run both, and run `apex report` on each run directory. For each report, read two things: the final mismatch, and whether the history is still falling at the last step or has flattened out. Of two attributes of the same kind, prefer the one that ends lower and is still improving. For example, if one history ends at a lower value and is still dropping while the other has stalled, the first fits better. Compare only attributes of the same kind. Among `max`, `min`, `mean`, `rms` and `sum_negative` the values are amplitudes, but they are different statistics, so read the shape of the history rather than treating the numbers as interchangeable. `trace` returns samples and `time_shift` returns a time, so their mismatch values are not on the same scale as the amplitude attributes and should not be compared with them. To check what an attribute computes, use the hand calculations below, which need no run.

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

The time shift is a shift in two-way time, and `time_shift` supplies its reference: the two-way time to the reservoir's base. It does not average or sum amplitudes, and it has no options, so you give it no value yourself. Choose it when you want that time rather than an amplitude from the window.

Because the result is a time and not a function of the amplitude statistics above, there is no sample arithmetic to check by hand. Treat it as a time, and do not compare it with the other attributes.

## Pitfalls

- **Expecting options.** These plugins have none. Their option sets accept no properties, so there is no threshold or parameter to adjust. A setting you add anyway is not valid for the plugin. Remove it and choose a different plugin to get different behaviour.
- **Looking for a window setting on the plugin.** The window is not an option of any of these plugins; it belongs to the run setup you started from.
- **Using `mean` to judge strength.** Positive and negative samples cancel in the mean. Use `rms` for overall energy.
- **Confusing `min` with `sum_negative`.** `min` is one value, the deepest trough. `sum_negative` adds up every negative sample, so it reflects all of them, not just the deepest.
- **Forgetting that `rms` is the default.** If you want a peak or trough, choose `max` or `min` explicitly.
- **Reading `trace` as a single number.** It returns samples. Inside the window they are unchanged, and outside it they are zero.
- **Mixing up `time_shift` with an amplitude attribute.** It returns a two-way time, so do not average it with, or plot it on the same scale as, amplitudes.
- **Comparing mismatch values across kinds of attribute.** Compare runs that differ only in the attribute, and only between attributes of the same kind.
- **Treating `sum_negative` as proof.** It is an indicator of soft sand or gas, not a confirmation. Compare it with other evidence.

## What to read next

- Run `uv run apex plugins` to see every plugin and its options, including those in other registries, and `uv run apex plugins --json` for the full descriptions in a form other tools can read.
- Run `uv run apex run config.toml --max-steps 1` on a small case, then continue with `uv run apex resume RUN_DIR`. If you need to pause a run that is going, `uv run apex stop RUN_DIR` asks it to stop at its next checkpoint, and `apex resume` continues it.
- Inspect a finished or partial run with `uv run apex report RUN_DIR`.
- Read the other plugin pages in this guide's plugins section, to see which processing steps you can combine with these attributes.
