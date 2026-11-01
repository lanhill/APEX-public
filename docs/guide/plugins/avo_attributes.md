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

APEX groups plugins of one kind into a registry, and `apex plugins` lists them by registry. The `avo_attributes` registry holds seven plugins: `max`, `mean`, `min`, `rms`, `sum_negative`, `time_shift` and `trace`. Most of them reduce the samples of a trace inside a window to one amplitude attribute. `trace` returns the windowed trace itself, and `time_shift` returns a two-way time instead of an amplitude.

In a history-matching run, the attribute you choose is the quantity computed from the model's data and compared with the data you matched against. A different attribute therefore gives a different mismatch.

This lesson explains what each plugin computes, how to choose between them, and how to compare two of them in a run.

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
- Access to the `apex` command line from the APEX folder.
- A working run configuration (the TOML file you pass to `apex run`).

The window is not an option of these plugins. It comes from the rest of your run configuration, which this lesson does not change. If you start from a configuration that already runs and change only the name of the attribute plugin, the window stays exactly as it was. The worked examples below simply assume a window is given.

## The plugins

None of these plugins has any options. Each one accepts nothing extra, so there is nothing to tune. You choose the behaviour by choosing the plugin.

| Plugin | What it returns over the window |
|---|---|
| `max` | The most positive amplitude (a peak). |
| `min` | The most negative amplitude (a trough). |
| `mean` | The mean amplitude. |
| `rms` | The root mean square amplitude, `sqrt(mean(s²))`. This is the default. |
| `sum_negative` | The sum of the negative amplitudes (a soft-sand or gas indicator). |
| `time_shift` | The two-way time to the reservoir's base, used for the time shift. |
| `trace` | The whole trace: the samples in the window, and zero outside it. |

## Steps

1. Decide the question you are asking of the window.
2. Pick the plugin:
   - Strongest positive reflection: `max`.
   - Strongest negative reflection: `min`.
   - Overall bias of the window: `mean`.
   - Overall energy, regardless of sign: `rms`. It is the default, so it is what you get if you do not choose another.
   - Evidence of soft sand or gas: `sum_negative`.
   - The two-way time to the reservoir's base: `time_shift`.
   - The samples themselves, not a single number: `trace`.
3. Check that the plugin is installed. Run:

   ```
   uv run apex plugins
   ```

   This lists every plugin with its options. Look for the `avo_attributes` group: it should show the name you picked, with no options. If the name is not listed, it is not available in your installation, so pick one that is. For the full descriptions as JSON, add the `--json` flag:

   ```
   uv run apex plugins --json
   ```

4. Open a run configuration that already works. Search the file for the name of the AVO attribute plugin it currently uses (for example `rms`) and replace only that name with the plugin you picked (for example `max`). Add no options, because these plugins have none. Leave everything else unchanged, including the window. Save the file under a new name for each attribute you try, such as `config_max.toml`, so the copies stay apart.
5. Start the run:

   ```
   uv run apex run config_max.toml
   ```

   To test on a small scale, stop early with `--max-steps`. An analysis step is one of the steps in which the run compares the model's data with the data you matched against and updates the model; `--max-steps 1` stops the run after the first one. You can continue later with `apex resume` and the directory of the run:

   ```
   uv run apex run config_max.toml --max-steps 1
   ```

   Progress is shown on standard error. Use `--no-progress` to turn it off (`--progress` is the default).
6. Look at the result of the run. `apex report` prints the data mismatch history and the performance metrics of a run. The data mismatch measures how far the data computed from the current model is from the data you matched against, and the history shows how it changes from one analysis step to the next. Give the command the directory of the run (`RUN_DIR` below stands for that directory):

   ```
   uv run apex report RUN_DIR
   ```

   Add `--plots` to also write the result figures as PNG and PDF. They go to `RUN_DIR/report` unless you give another directory with `--output`:

   ```
   uv run apex report RUN_DIR --plots --output figures
   ```

7. To compare two plugins, make one copy of the configuration per plugin, change only the attribute plugin in each copy, and run both. Keep a note of which run directory belongs to which configuration, then run `apex report` on each. Read two things in each report: the final mismatch, and whether the history is still falling at the last step or has flattened out. A history that keeps dropping from step to step is still improving; one that stays at about the same value over the last steps has levelled off.

   The amplitude attributes `max`, `min`, `mean`, `rms` and `sum_negative` are different statistics of the same samples, so their mismatch values are not interchangeable numbers. Compare the shape of the histories, not the raw values. `trace` returns samples and `time_shift` returns a time, so do not compare their mismatch values with those of the amplitude attributes at all.

   For example, if the history of the `max` run is still falling at its last step while the `rms` run levelled off at an early step, the `max` run is still improving and has more to give. This says nothing about whether `max` is the better attribute in absolute terms. To check what an attribute computes, use the hand calculations below, which need no run.

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

`time_shift` returns the two-way time to the reservoir's base, which is the time used for the time shift. It does not average or sum amplitudes, and it has no options, so you give it no value yourself.

Choose it when your question is about *when* the reservoir's base appears in two-way time, rather than *how strong* the amplitude is. For example, "how deep in time is the base of the reservoir?" is a question for `time_shift`, while "is the reflection from the reservoir stronger?" is a question for `max` or `rms`.

Because the result is a time and not a sample statistic, there is no sample arithmetic to check by hand. Do not compare it, or its mismatch, with the amplitude attributes.

## Pitfalls

- **Expecting options.** These plugins have none. Their option sets accept no properties, so there is no threshold or parameter to adjust. A setting you add anyway is not valid for the plugin. Remove it and choose a different plugin to get different behaviour.
- **Looking for a window setting on the plugin.** The window is not an option of any of these plugins. It comes from the rest of the configuration.
- **Using `mean` to judge strength.** Positive and negative samples cancel in the mean. Use `rms` for overall energy.
- **Confusing `min` with `sum_negative`.** `min` is one value, the deepest trough. `sum_negative` adds up every negative sample, so it reflects all of them, not just the deepest.
- **Forgetting that `rms` is the default.** If you want a peak or trough, choose `max` or `min` explicitly.
- **Reading `trace` as a single number.** It returns samples. Inside the window they are unchanged, and outside it they are zero.
- **Mixing up `time_shift` with an amplitude attribute.** It returns a two-way time, so do not average it with, or plot it on the same scale as, amplitudes.
- **Comparing mismatch values across attributes.** Runs that differ in the attribute have different statistics behind their mismatch. Compare the shape of the histories, and never compare `trace` or `time_shift` runs with amplitude runs.
- **Letting two runs get mixed up.** Give each configuration copy its own name and write down which run directory came from which.
- **Treating `sum_negative` as proof.** It is an indicator of soft sand or gas, not a confirmation. Compare it with other evidence.

## What to read next

- Run `uv run apex plugins` to see every plugin and its options in all registries, and `uv run apex plugins --json` for the full descriptions in a form other tools can read.
- Run `uv run apex run config_max.toml --max-steps 1` on a small case, then continue with `uv run apex resume RUN_DIR`. To pause a run that is going, `uv run apex stop RUN_DIR` asks it to stop at its next checkpoint, and `uv run apex resume RUN_DIR` continues it.
- Inspect a finished or partial run with `uv run apex report RUN_DIR`.
