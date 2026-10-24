---
audience: public
generated: reference
---

# The apex grant commands: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex grant create`

Approve up to a budget of one client's experiment ahead, until it expires.

```bash
apex grant create CLIENT EXPERIMENT
```

| option | default | description |
|---|---|---|
| `--max-simulator-runs` | | The total budget of simulator runs. |
| `--expires` | | How long it lasts: 7d, 12h or 30m (at most 30d). |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex grant list`

List the active grants: id, client, experiment, used/budget, expiry.

```bash
apex grant list
```

| option | default | description |
|---|---|---|
| `--all` | false | Also the expired and revoked grants. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex grant revoke`

Revoke a grant at once (submissions already let through are not stopped).

```bash
apex grant revoke GRANT_ID
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
