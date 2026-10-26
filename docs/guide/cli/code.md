---
audience: public
items:
  - cli:apex code freeze
  - cli:apex code list
  - cli:apex code remove
---

# The `apex code` commands

This lesson shows you how to manage frozen copies of your code with the three `apex code` commands: `apex code freeze`, `apex code list` and `apex code remove`.

## Goal

By the end you will be able to:

- make a frozen copy of a commit with `apex code freeze`,
- see which frozen copies exist, and what uses each, with `apex code list`,
- delete a copy you no longer need with `apex code remove`.

A frozen copy is a snapshot of a commit, kept in its own folder, so that work started from that commit keeps using the same code even if the repository moves on. Work that runs from a copy (a job or a process) is said to *use* it, and a copy that something uses cannot be removed.

## Prerequisites

- `apex` installed and available on your command line.
- A Git repository with at least one commit. Run the commands from inside it.
- Optionally, a services root: the folder where frozen copies are kept. By default it is `$APEX_HOME`, otherwise `~/.apex/service`. All three commands accept `--root` to point somewhere else.

## Steps

### 1. Freeze a commit: `apex code freeze`

```bash
uv run apex code freeze [commit] [--root ROOT]
```

- `commit` is optional. It can be a commit, a branch or a tag. The default is `HEAD`.
- `--root` sets the services root.

The command makes the frozen copy of that commit and prints the copy's folder. That folder is inside the services root.

A copy can be left unfinished, for example if an earlier `freeze` was interrupted. You do not need to do anything special: run the same `apex code freeze` command again and it completes the copy, then prints the folder.

```bash
uv run apex code freeze            # freezes HEAD
uv run apex code freeze main       # freezes the tip of a branch
uv run apex code freeze v1.0       # freezes a tag
```

### 2. List the frozen copies: `apex code list`

```bash
uv run apex code list [--root ROOT]
```

This lists every frozen copy and shows what uses each one. Each copy is identified by its 12-character commit, a short form of the commit's ID. This is the value you give to `apex code remove`. Read the listing like this:

- A copy that something uses (a job or a process) shows what uses it. Do not expect to be able to remove it.
- A copy that shows nothing using it can be removed.

Use the listing before removing anything.

### 3. Remove a frozen copy: `apex code remove`

```bash
uv run apex code remove COMMIT [--root ROOT]
```

- `COMMIT` is required. It is the copy's 12-character commit, as shown by `apex code list`.
- `--root` sets the services root.

The command refuses to remove a copy while a job or a process uses it. In that case the copy is left in place.

## Worked example

Freeze the current commit. The command prints the copy's folder; note it:

```bash
uv run apex code freeze
```

List the copies to check that yours is there and to see what uses it:

```bash
uv run apex code list
```

Find your copy in the listing and copy its 12-character commit. If the listing shows that a job or a process still uses it, wait until that work has finished and list again. When nothing uses the copy any more, remove it (the value below is only an illustration; use one from your own listing):

```bash
uv run apex code remove 0123456789ab
```

If the copy is still in use, the command refuses and the copy stays. If it succeeds, run `apex code list` again and the copy is no longer shown.

If you keep your services somewhere other than the default, pass the same `--root` to every command:

```bash
uv run apex code freeze main --root /data/apex-services
uv run apex code list --root /data/apex-services
uv run apex code remove 0123456789ab --root /data/apex-services
```

## Pitfalls

- **Removal is refused while the copy is in use.** If a job or a process uses the copy, `apex code remove` will not delete it. Check `apex code list` to see what uses it, and try again once it is finished.
- **Use the 12-character commit for removal.** `apex code remove` takes the copy's 12-character commit from `apex code list`, not a branch or tag name.
- **Keep `--root` consistent.** Each command works on one services root. If you freeze with a custom `--root` but list without it, you will look at a different root and the copy will not appear.
- **`freeze` defaults to `HEAD`.** Without an argument it freezes the commit you currently have checked out. Name a branch, tag or commit if you want something else.
- **An interrupted freeze is not lost work.** Run the same `apex code freeze` command again to complete the copy.

## What to read next

Continue with the other pages of the command-line guide (under `guide/cli/`) to see how jobs and processes use your frozen copies.
