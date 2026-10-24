---
audience: public
items:
  - cli:apex runs list
  - cli:apex runs show
---

# The apex runs commands

## Goal

APEX keeps a catalogue of the runs it has performed. In this lesson you will use two commands to work with that catalogue:

- `apex runs list` lists the catalogued runs, oldest first.
- `apex runs show` shows the summary of one run, as JSON.

By the end you will be able to find a run, narrow the list by annotation, and read one run's summary.

## Prerequisites

- APEX is installed and the `apex` command is available in your shell.
- At least one run has been catalogued under your services root. If the catalogue is empty, `apex runs list` has nothing to show.
- You know where your services root is. By default it is `$APEX_HOME` if that variable is set, otherwise `~/.apex/service`. You only need the `--root` option if your runs live somewhere else.

## Steps

### 1. List the runs

```bash
apex runs list
```

The runs are printed oldest first, so the most recent run is at the end of the output.

### 2. Narrow the list by annotation

If your runs carry annotations, `--annotation` keeps only the runs with a given annotation. Give it a `key=value` pair:

```bash
apex runs list --annotation key=value
```

Replace `key` and `value` with an annotation your runs actually have.

### 3. Show one run

`apex runs show` takes a required `run` argument: a run id, or its first characters. A short prefix is enough as long as it identifies the run.

```bash
apex runs show <run>
```

The summary is printed as JSON.

### 4. Use a different services root

Both commands accept `--root`. It points them at another services root instead of `$APEX_HOME` or `~/.apex/service`:

```bash
apex runs list --root /path/to/service
apex runs show <run> --root /path/to/service
```

## Worked example

Suppose you want to inspect your most recent run.

1. List the runs:

   ```bash
   apex runs list
   ```

   The list is oldest first, so the last entry is the newest. Note its run id.

2. Show that run, using the id or just its first characters. Here `1a2b3c` stands for the start of your run id:

   ```bash
   apex runs show 1a2b3c
   ```

   The command prints the run's summary as JSON.

3. If you only care about runs with a certain annotation, repeat step 1 with a filter, then show the one you want:

   ```bash
   apex runs list --annotation key=value
   ```

## Pitfalls

- **Order.** `apex runs list` is oldest first. The run you just made is at the end, not the start.
- **Annotation format.** `--annotation` expects `key=value`. Other forms will not match what you intend.
- **Ambiguous prefix.** For `apex runs show`, a prefix of the run id works only if it identifies the run. If it does not, use more characters of the id.
- **Empty or unexpected results.** If the list is empty or a run is missing, check that you are looking at the right services root. Set `--root` or `$APEX_HOME` as needed.
- **Forgetting `--root` on the second command.** If you used `--root` with `apex runs list`, use the same value with `apex runs show`.

## What to read next

The rest of the command-line guide covers the other `apex` commands. Return to this page whenever you need to find a run and read its summary.
