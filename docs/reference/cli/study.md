---
audience: public
generated: reference
---

# The apex study commands: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex study compare`

Compare a study's variants at each run's posterior.

```bash
uv run apex study compare NAME
```

| option | default | description |
|---|---|---|
| `--metric` | | 'metric:label'; repeatable. |
| `--json` | false | The comparison, as JSON. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex study run`

Validate and submit a study; its runs go on in the background.

```bash
uv run apex study run STUDY
```

| option | default | description |
|---|---|---|
| `--yes` | false | Approve a large study without asking. |
| `--wait` (on) / `--no-wait` (off) | false | Wait for the study to finish. |
| `--max-concurrent-jobs` | | Jobs at once (default: service.toml, else 2). |
| `--priority` | normal | high, normal or low. |
| `--approval-token` | | A token approved with `apex approve`. |
| `--code` | | live, frozen (the last commit, for every cell) or a commit (default: [jobs] cli_code, frozen). |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex study status`

Show a study's state and every run's.

```bash
uv run apex study status NAME
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
