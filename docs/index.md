---
audience: public
---

# APEX

APEX (Agentic Platform for Ensemble eXploration) is a platform for ensemble history matching, reservoir forecasting, uncertainty quantification, optimization, and agent-driven decision support.

This page is the map of the documentation. Use it to find the page that fits what you want to do.

## Start here

New to APEX? Read the [User guide](guide/index.md) first, then try [Examples and toy problems](examples.md). When you have run a small example, come back to this page and pick the topic you need from the lists below.

Three step-by-step pages get you going:

- [The web UI and its AI assistant](getting_started/web_ui.md): start the web UI, run an example from it, and give the assistant a model (Claude through Claude Code, or Gemma through Ollama).
- [The agents from the command line](getting_started/agents_cli.md): the monitor, diagnosis, planner, report and plotter in a terminal, with the same models, and Claude Code driving APEX's tools.
- [Write and register your own plugin](getting_started/writing_a_plugin.md): a new metric and a new history-matching algorithm, in packages of your own.

## Core topics

- [User guide](guide/index.md): the main lessons on using APEX. Begin here.
- [Examples and toy problems](examples.md): small problems to run and learn from.
- [Optimization](optimization.md): how APEX is used for optimization.
- [GMAC and C-GIES](gmac.md): the GMAC and C-GIES methods, described on one page.
- [ERT](ert.md): how APEX relates to and works with ERT.
- [Run summaries](summaries.md): how the results of a run are summarized.

## Services, interfaces and agents

- [Job service](service.md): the service that runs APEX jobs.
- [REST API](api.md): the HTTP interface to the job service.
- [Tools for LLMs](tools.md): the tools that let language-model agents use APEX.
- [Web UI](ui.md): the browser interface.
- [Runtime agents](agents.md): the agents that work with APEX while it runs.

## Migration and case study

- [PET migration status](migration/pet_parity.md): PET is the earlier tool APEX is migrating from. This page tracks how far that migration has come. Read it if you have used PET before.

## Which page should I read?

- You want to learn APEX from scratch: [User guide](guide/index.md), then [Examples and toy problems](examples.md).
- You want to call APEX from your own code or from an agent: [Job service](service.md), [REST API](api.md) and [tools for LLMs](tools.md).
- You prefer a graphical interface: [Web UI](ui.md).
- You are coming from PET: [PET migration status](migration/pet_parity.md).
