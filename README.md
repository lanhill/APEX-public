# APEX

**Agentic Platform for Ensemble eXploration**: a framework for ensemble-based history matching, forecasting, uncertainty quantification and optimization of subsurface models, with agents that plan, monitor and report on the work. It natively re-implements and extends NORCE's [Python Ensemble Toolbox (PET)](https://github.com/Python-Ensemble-Toolbox/PET).

APEX is compatible with ERT: with APEX's interface, each can run the other's cases ([APEX and ERT](docs/ert.md)). PET and ERT cases convert to and from APEX cases, within what each can express ([the case-converter](docs/agents.md#bringing-your-own-case-the-case-converter)).

Status: early development.

## Quick start

<table>
<tr>
<td width="50%"><a href="docs/getting_started/first_run.md">Your first history-matching run, in a terminal</a><br>▶ <a href="videos/ep02_first_case/ep02_first_case.mp4">Ep. 2</a> · <a href="videos/ep03_cost_and_first_steps/ep03_cost_and_first_steps.mp4">3</a> · <a href="videos/ep04_resume_and_results/ep04_resume_and_results.mp4">4</a> · <a href="videos/ep05_figures_and_comparison/ep05_figures_and_comparison.mp4">5</a></td>
<td width="50%"><a href="docs/getting_started/studies_and_approvals.md">Studies, the job service and approvals</a><br>▶ <a href="videos/ep08_hm_study/ep08_hm_study.mp4">Ep. 8</a> · <a href="videos/ep09_job_service/ep09_job_service.mp4">9</a> · <a href="videos/ep10_approvals/ep10_approvals.mp4">10</a></td>
</tr>
<tr>
<td width="50%"><a href="docs/getting_started/agents_cli.md">Using agents from the command line</a><br>▶ <a href="videos/ep12_agents_cli/ep12_agents_cli.mp4">Ep. 12</a></td>
<td width="50%"><a href="docs/getting_started/case_converter.md">Bringing your own case to APEX: one deck or one description</a><br>▶ <a href="videos/ep13_own_case/ep13_own_case.mp4">Ep. 13</a></td>
</tr>
<tr>
<td width="50%"><a href="docs/getting_started/web_ui.md">Setting up your web UI with an AI assistant</a><br>▶ <a href="videos/ep11_web_ui/ep11_web_ui.mp4">Ep. 11</a></td>
<td width="50%"><a href="docs/getting_started/writing_a_plugin.md">Writing and registering your own plugin</a><br>▶ <a href="videos/ep14_plugin_metric/ep14_plugin_metric.mp4">Ep. 14</a> · <a href="videos/ep15_plugin_algorithm/ep15_plugin_algorithm.mp4">15</a></td>
</tr>
</table>

▶ marks the videos of each page. [All the videos](videos/README.md), with subtitles, include a two-minute overview, the installation and optimization.

## Installation

APEX is tested on Linux with Python 3.12. The steps below take a few minutes; only OPM Flow (step 5) needs administrator rights, and only for reservoir simulations. Click a step to open or close it.

<details open>
<summary><b>1. Install the prerequisites</b></summary>

You need `git` and [uv](https://docs.astral.sh/uv/), the Python package manager APEX uses. uv also downloads Python 3.12 if your system lacks it (the repository's `.python-version` asks for 3.12).

```bash
git --version                                     # any recent git
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv into ~/.local/bin
```

Open a new terminal afterwards (or run `source ~/.local/bin/env`) so that `uv` is on your `PATH`, and check it with `uv --version`.

</details>

<details open>
<summary><b>2. Get the code</b></summary>

```bash
git clone https://github.com/lanhill/APEX-public.git apex
cd apex
```

</details>

<details open>
<summary><b>3. Create the environment</b></summary>

```bash
uv sync --locked
```

This creates `.venv/` inside the folder with APEX, its exact pinned dependencies (`uv.lock`), the development and documentation tools, and most optional features: reservoir simulators' files, figures, geostatistical and facies priors, the REST API and web UI server, MCP, the runtime agents' model providers and eCalc. Outside the folder, uv keeps its download cache in `~/.cache/uv`, its own program in `~/.local/bin` (step 1), and a Python it downloads in `~/.local/share/uv/python`. Every later command is run from this folder with `uv run`, which uses that environment.

APEX's own messages and the web UI write commands as a plain `apex …`. To type them as they are, activate the environment in that terminal first; `apex` then works without `uv run` until the terminal closes (or `deactivate`):

```bash
source .venv/bin/activate
```

</details>

<details open>
<summary><b>4. Check the installation</b></summary>

```bash
uv run apex --help        # the command-line interface
uv run apex plugins       # every plugin you can choose (methods, simulators, priors, the agents' skills, ...); [parity], [validated], [stable], [experimental] tags show a status
uv run pytest -q -n 4     # the test suite, on 4 cores
```

</details>

<details>
<summary><b>5. Add what your work needs (optional)</b></summary>

Three optional features are not in the default environment. Name the ones you need, all in one command; each `uv sync` installs exactly what it is given, so repeat earlier ones when adding another:

```bash
uv sync --locked --extra co2 --extra ert
```

| Extra | For |
|---|---|
| `co2` | CO₂'s equation of state (CoolProp), for monitoring CO₂ storage |
| `ert` | the interface to Equinor's ERT |
| `torch` | PyTorch (CPU build), for your own scripts; APEX itself does not need it |

**Reservoir simulations** also need [OPM Flow](https://opm-project.org), which is not a Python package. On Ubuntu 24.04 or later (OPM's packages):

```bash
sudo apt-get install software-properties-common   # provides apt-add-repository
sudo apt-add-repository ppa:opm/ppa
sudo apt-get update
sudo apt-get install mpi-default-bin libopm-simulators-bin
flow --version
```

For other systems, see OPM's [binary packages](https://opm-project.org/?page_id=245). Commercial simulators (Eclipse) are used when installed and licensed on your machine.

**The web UI** is a separate front end, built with Node.js 22.12 or later (Node 18.18 still builds it, with npm's engine warnings; its tests need 22):

```bash
npm --prefix ui ci && npm --prefix ui run build
uv run apex service start --http 127.0.0.1:8765   # runs in the background; then open http://127.0.0.1:8765/ui/
uv run apex service token --ui --owner            # prints the token to sign in with
uv run apex service stop                          # when you are done
```

</details>

<details>
<summary><b>6. Run a first example</b></summary>

A history-matching run on a small linear-Gaussian case, which needs no simulator. [Your first history-matching run](docs/getting_started/first_run.md) walks through it step by step: a copy of the example, its configuration, validating, stopping and resuming, the results and a second method.

```bash
uv run python examples/linear_gaussian/make_case.py
uv run apex run examples/linear_gaussian/config_esmda.toml
uv run apex report examples/linear_gaussian/runs/linear_esmda          # its history (runs go next to their configuration)
uv run apex report examples/linear_gaussian/runs/linear_esmda --plots  # and figures, in its report/ folder
```

A run's folder (`runs/<name>` next to the configuration, `<name>` from its `[run] name`, unless `[run] output_dir` names another) holds:

- its configuration as run (`config.json`), its state, history and summary (`state.json`, `history.json`, `summary.json`, `events.jsonl`) and its ensembles (`checkpoints/`);
- `provenance.json` (the APEX commit, the libraries, the host and the inputs' hashes that made it) and `snapshot_version` (the format of the folder, for reading older runs);
- `run.lock`, the file a run locks while it writes. It stays in the folder after the run and is harmless: the lock goes with the process, however the process ends.

`uv run apex schema KIND` prints the JSON Schema of what a configuration file may hold, for each kind: `history_match`, `prediction`, `study`, `optimization`, `optimization_study` and `service`.

Running the same configuration again is refused while that folder holds a run: resume it with `uv run apex resume <folder>`, or give the run another folder (another `[run] name`, or `[run] output_dir`).

With OPM Flow, `uv run apex run examples/3dbox_production_avo/config_esmda.toml` history-matches the 3dBox case on its rates and seismic maps (500 OPM Flow simulations: hours on a workstation). The `examples/` folder has more; most are described by their `example.toml`.

</details>

<details>
<summary><b>7. Read the documentation</b></summary>

```bash
uv run mkdocs serve       # then open http://127.0.0.1:8000
```

It describes the plugins, the configuration files and the workflows, with a user guide and notebooks. Its **Getting started** section has six step-by-step pages:

- *Your first history-matching run*: the linear-Gaussian example in a terminal, from a copy of it to its summary, report and figures.
- *The web UI and its AI assistant*: run an example from the browser, and give the assistant a model (Claude through Claude Code with your subscription, or Gemma through Ollama).
- *The agents from the command line*: the monitor, diagnosis, planner, report and plotter in a terminal, and Claude Code driving APEX's tools.
- *Bringing your own case*: the case-converter and the geomodeler turn a deck and its history into an APEX case, give it a prior, and create a synthetic case from a description.
- *Studies, the job service and approvals*: compare two methods over seeds in a study, run the job service, and approve a submission over the cost limits.
- *Write and register your own plugin*: a new performance metric and a new history-matching algorithm, each in a package of your own.

</details>

<details>
<summary><b>Updating</b></summary>

```bash
git pull
uv sync --locked          # with the same --extra options as before
```

</details>

<details>
<summary><b>Using APEX from another project</b></summary>

APEX can also be a dependency of your own project; name the optional features you need:

```bash
uv add "apex[reservoir,plots] @ git+https://github.com/lanhill/APEX-public.git"
```

Its features are then available as `import apex` (for example `apex.client.Client`, the same calls as the command line), and its command line as `uv run apex`. The web UI is not included this way. (An unrelated package on PyPI is also called `apex`; installing from this address avoids it.)

</details>

## Development

<details>
<summary><b>The checks and the tests</b></summary>

```bash
uv run pre-commit install    # format, lint, type and architecture checks on every commit
uv run ruff format . && uv run ruff check . && uv run mypy && uv run lint-imports && uv run pytest
```

`pytest -n 4` turns the performance benchmarks (pytest-benchmark) off: run them on their own, without `-n`, as `uv run pytest --benchmark-only`.

The parity tests compare against stored reference results; `tests/data/reference/README.md` says how each was made and how to set up the environments that regenerate them.

</details>

## Disclaimer

<details open>
<summary><b>The disclaimer</b></summary>

This repository is made available and contributed to under a licence whose terms, for the protection of the contributors, make clear that it is offered "as is", without warranty, and disclaim liability for damages resulting from its use. APEX's licence, the GNU General Public License version 3 or later, includes such terms (sections 15 and 16).

The code may include mistakes, and cannot address every situation. If there is any question, we encourage you to do your own research, discuss with your community or contact us through the repository's issues.

All views expressed here are those of the contributors, and do not represent the opinions of any entities with which the contributors have been, are now or will be affiliated.

</details>

## Licence

GPL-3.0-or-later. APEX adapts algorithms from PET (GPL-3.0). GIES, its localization methods, the performance metrics, model error characterization and the classic figures are ported from PET-fork, a NORCE fork of PET (not public), at commit 1348861. Their publication here under GPL-3.0-or-later is cleared with the fork's owner.
