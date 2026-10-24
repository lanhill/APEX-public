---
audience: public
generated: reference
---

# The REST API: reference

Every one of these 51 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `DELETE /v1/chats/{chat_id}`

Delete Chat



## `DELETE /v1/files/content`

Delete File

Delete a file of one's own web-UI folder, if unchanged since read.

## `GET /v1/agents`

Agents

Every agent with its visibility; a guest gets the public ones its list allows.

## `GET /v1/agents/plot/{run_id}`

Plot State

Return how a run's figure drawing stands: none, drawing, done or failed.

## `GET /v1/chats`

Chats



## `GET /v1/chats/{chat_id}`

Chat



## `GET /v1/chats/{chat_id}/events`

Chat Events



## `GET /v1/deletions`

Deletion Preview

Return what deleting a job (or a run's, study's or prediction's submission) removes.

## `GET /v1/events`

Feed



## `GET /v1/examples`

Examples

Return the catalogue of examples and toy problems, each with its workflows.

## `GET /v1/files`

Files



## `GET /v1/files/content`

File Content



## `GET /v1/health`

Health



## `GET /v1/jobs`

Jobs



## `GET /v1/jobs/{job_id}`

Job



## `GET /v1/jobs/{job_id}/events`

Job Events



## `GET /v1/optimizations`

Optimizations



## `GET /v1/optimizations/summary`

Optimization Summary



## `GET /v1/plugins`

Plugins



## `GET /v1/predictions/summary`

Prediction Summary



## `GET /v1/runs`

Runs



## `GET /v1/runs/{run_id}`

Run



## `GET /v1/runs/{run_id}/artifacts/{path}`

Artifact



## `GET /v1/runs/{run_id}/figures`

Figures

List the figures `apex report --plots` wrote (PNG, under `report/`).

## `GET /v1/runs/{run_id}/steps`

Run Steps



## `GET /v1/schemas/{kind}`

Schema



## `GET /v1/studies/{name}`

Study



## `GET /v1/studies/{name}/comparison`

Comparison



## `GET /v1/ui/pages`

Contributed Pages

Return the web-UI pages installed packages add that the caller may see.

## `GET /v1/whoami`

Whoami

Return the token's role, and what a UI user may do.

## `GET /v1/workspace`

Workspace

Return the roots: each one's name (the prefix of relative paths) and folder.

UI users get the names only: every route takes relative paths.

## `POST /v1/agents/diagnose`

Diagnose

Return a diagnosis of a run: causes, with evidence, and recommended changes.

## `POST /v1/agents/monitor`

Monitor

Return the run monitor's report on a run (advisory: it never stops anything).

## `POST /v1/agents/plan`

Plan

Plan a study for a case and a goal, as a draft in the user's web-UI folder.

## `POST /v1/agents/plot`

Plot

Choose a run's figures and draw them in the background into <run>/report/plots.

A classic set takes minutes: the answer comes once the figures are chosen, and
they appear under GET /v1/runs/{id}/figures as they are drawn (`summary.json`
says when all are).

## `POST /v1/agents/report`

Report

Return the narrative report of a run or a study (Markdown; numbers checked).

## `POST /v1/chats`

Create Chat



## `POST /v1/chats/{chat_id}/messages`

Message



## `POST /v1/chats/{chat_id}/proposals/{n}:discard`

Discard



## `POST /v1/chats/{chat_id}/proposals/{n}:launched`

Launched

Record that the user launched a proposal (the UI submitted it) as a job.

## `POST /v1/chats/{chat_id}:stop`

Stop Chat

Stop the chat's running answer at once; the user's own chat.

`stopped` is false when nothing was running (it had just finished). Nothing done
before the stop is undone, and the chat takes a new message at once.

## `POST /v1/configs:validate`

Validate



## `POST /v1/deletions`

Deletion Confirmed

Delete one's own submission, as the preview's token describes it.

## `POST /v1/examples:copy`

Copy Example

Copy an example into one's own UI folder; return where its configurations are.

## `POST /v1/jobs/{job_id}:resume`

Resume



## `POST /v1/jobs/{job_id}:stop`

Stop



## `POST /v1/optimizations`

Submit Optimization



## `POST /v1/predictions`

Submit Prediction



## `POST /v1/runs`

Submit Run



## `POST /v1/studies`

Submit Study



## `PUT /v1/files/content`

Save File
