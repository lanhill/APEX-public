---
audience: public
---

# The web UI

The web UI is a React app served by the job service. It calls only the REST API (see [the REST API](api.md)), so everything it shows is what `apex` and the tools would show. New to it? [Getting started with the web UI](getting_started/web_ui.md) walks through it step by step, the assistant's model included.

## Starting it

Build it once (Node 22.12 or later; Node 18.18 still builds it, with npm's engine warnings, but its tests need 22), on any machine that has the repository, then start the service with the API:

```bash
npm --prefix ui ci && npm --prefix ui run build   # writes ui/dist (not committed)
uv run apex service start --http 127.0.0.1:8765
uv run apex service token --ui --owner                   # your web-UI token, to paste into the sign-in page
```

Open `http://127.0.0.1:8765/ui/`. From another machine, tunnel the port (`ssh -L 8765:127.0.0.1:8765 host`). The token is kept for the browser session only. Sign in with the web-UI token (`--ui --owner`), not the API token that `apex service token` alone prints, which the web UI refuses. The web-UI token expires after `[ui] token_days` (7); `apex service token --ui --owner --rotate` issues a new one.

The service finds the build in `[ui] dist_dir` of `service.toml`, else `$APEX_UI_DIST`, else `ui/dist` beside the source tree. Without a build, `/ui/` answers with the command above, and the service's log says so at start (`no web UI build ...: set [ui] dist_dir`); with one, it logs where the UI is served from. The `service.toml` that counts is the one in the service's own root (`$APEX_HOME`).

## Using it from another machine (e.g. a Windows laptop)

Keep the service bound to `127.0.0.1` and reach it through an SSH tunnel. The tunnel carries the laptop's `localhost:8765` to the service over your normal SSH login, so nothing is exposed to the network. The laptop needs only a browser and an SSH client. Below, `you@server` is your SSH login on the machine that runs the service.

### 1. Open the tunnel

**Windows' built-in OpenSSH** (Windows 10/11), in PowerShell or Windows Terminal:

```powershell
ssh -N -L 8765:127.0.0.1:8765 you@server
```

- `-N` opens no shell; the window just holds the tunnel. Keep it open while you use the UI, and press Ctrl+C to close it.
- If `ssh` is missing, enable it under Settings → System → Optional features → OpenSSH Client.
- A shortcut in `C:\Users\<you>\.ssh\config` makes it `ssh -N apex`:

  ```
  Host apex
      HostName server
      User you
      LocalForward 8765 127.0.0.1:8765
  ```

  With an SSH key (`ssh-keygen`, then the public key in the server's `~/.ssh/authorized_keys`), no password is asked.

**PuTTY:**
1. Session: the server's host name, port 22.
2. Connection → SSH → Tunnels: source port `8765`, destination `127.0.0.1:8765`, **Add**.
3. Save the session, then **Open** and log in. Keep the window open.

**VS Code Remote-SSH:** in a remote window, open the **Ports** panel, click **Forward a Port** and enter `8765`. VS Code keeps the tunnel open while the remote window is open.

On Linux or macOS, the OpenSSH command above works as it is.

### 2. Sign in

1. Open `http://localhost:8765/ui/` in the laptop's browser.
2. Enter your **user name** and your **token**:
   - **The owner:** the name `owner`, and the web-UI token `apex service token --ui --owner` prints on the server. Older services, without web-UI tokens for the owner, take the API token of `apex service token`.
   - **A guest:** the name and the token that APEX's service gave you (the owner issues them with `apex service token --ui --guest NAME`).

   The name must be the token's own, or the sign-in is refused.
3. **Keep me signed in on this computer for 7 days** is optional. Without it, the sign-in lasts for that tab only, and closing the tab signs you out. With it, the token is stored in this browser until then, and never past the token's own expiry. Use it only on a computer you alone use. *Sign out* ends the sign-in, in every tab, but this browser keeps your name and token until the date, and the sign-in page says until when: the form comes back filled in, and one click signs you in again (each such sign-in starts a new 7 days, never past the token's own expiry). **Forget me on this computer**, on the sign-in page, removes them at once; a token the service refuses (expired or replaced) is removed by itself when it is next used. A password manager may also offer to save the token; what it keeps is outside these limits.
4. **When a token expires**, or the service stops accepting it (e.g. after `--rotate`), the page signs you out at its next request and says why, adding the service's explanation for an expired token or for the API token used where a web-UI token is needed. Ask the owner for a new token, then sign in again.

   The sign-in page's hint speaks to guests only; how the owner signs in is described here, not on the page.

If the port is busy on the laptop, forward another one, e.g. `-L 9000:127.0.0.1:8765`, and open `http://localhost:9000/ui/`.

### 3. Things to know

- **Paths are the server's.** The editor, the runs and every file path are on the server. Nothing is stored on the laptop, and the editor cannot open the laptop's files. Copy them to the server first (`scp`, WinSCP, or VS Code), into a workspace root.
- **Approval happens on the server.** When a submission needs approval, run the `apex approve TOKEN` it shows in an SSH session on the server, then click **Approved: submit again**.
- **If the page does not load:**
  - check that the tunnel is still open;
  - on the server, run `apex service status`;
  - start the service with `apex service start --http 127.0.0.1:8765` if it is not running.
- **Security.**
  - The tunnel and the token are the protection. The token travels only inside the encrypted tunnel.
  - Do not bind the service to another address to skip the tunnel, unless a reverse proxy terminates TLS in front of it.
  - After sharing a screen or a machine, `apex service token --ui --owner --rotate` replaces your web-UI token (and `apex service token --rotate` the API token). The old tab is refused at its next request, and you sign in again.
- **Other clients:** the same tunnel serves the Python client (`Client.connect("http://localhost:8765", token)`) and MCP hosts (`http://localhost:8765/mcp`; see [the tools page](tools.md)).

## Who sees what

Your token (`apex service token`) sees everything. To share the UI, give each guest a **token of their own**:

```bash
uv run apex service token --ui --guest alice      # created on first use; --rotate replaces it
uv run apex service token --list-guests
uv run apex service token --ui --guest alice --revoke   # refused at once
```

(`apex service token --ui` alone is the guest `guest`.)

- **A guest sees only their own work:** the jobs, runs, studies and predictions they submitted, and their own chats. They never see yours, or another guest's.
  - The server marks each of their submissions with `submitted_by = guest:<name>`, and a request cannot change it.
  - A study's cells and runs carry the mark too.
- **What they may open and use** is an allowlist in `<root>/ui_access.toml`: what it does not name is hidden. A file or folder counts where it really lies: a symlink in a listed folder (or in a guest's own) that leads into a hidden folder, or into another user's UI folder, is hidden like that folder, and left out of listings. A listed folder that is itself a link opens only if where it leads is listed too. Other users' UI folders stay hidden whatever the list says (even `folders = ["*"]`); a guest's own is always theirs. Private plugins, built for one site, are hidden from guests whatever the list says too.

```toml
[workspace]
folders = ["5Spots", "Brugge/*/configs"]   # workspace-relative folders, with their contents

[plugins]                             # per registry; "*" for all of one
methods = ["esmda", "gies"]
update_strategies = ["*"]
localizations = ["adaptive", "distance"]
simulators = ["opm_flow"]
executors = ["*"]
priors = ["*"]
metrics = ["*"]

[agents]
names = ["monitor", "diagnosis", "planner", "report"]

[actions]                             # what guests may do; all off by default but delete
submit = true                         # validate and submit files of the listed folders
stop = false                          # (their own jobs)
resume = false
edit = false                          # save files in the editor (their chats' drafts: always)
delete = true                         # delete their own finished work, after confirming

[programs]                            # programs their configurations may name; none by default
executables = ["flow"]                # bare names only
```

The `[plugins]` lists of `converter_skills`, `geomodeling_skills`, `variogram_estimators` and `chop_components` only choose what the Plugins page shows a guest. They do not limit what a guest may use. The case-converter and the geomodeler use their skills and estimators for guests as for you, within the agents' own limits (the agent listed in `[agents] names`, the web UI's size caps, writing only in the guest's folder; the converter's private skills excepted), and a CHOP run's components are its method's options, which the method's own entry governs.

- **Files and programs** (with or without a `[policy]`): a guest's configuration may name input, template and source files only in their own UI folder, a listed folder or one of their runs, judged by the file's real path (a symlink into another folder does not count as being in the listed one). A deck template is Python (Mako), so it must also not be text a guest can write or have written from what they type: never a `.toml`, `.md` or `.json` file (nor `.jsonl`, `.txt`, `.log`, `.csv`), nothing in a run, and in their own UI folder only a deck the assistant drafted (`drafts/`). A program must be a bare name in `[programs] executables`. Program arguments, plugin options that choose where to write, and options a plugin does not declare are refused. Validation says which; a submission answers *not allowed*. The assistant's drafts are held to the same rules (an optimization drafted from a case only if the case's own files pass), and another user's UI folder never counts as a listed folder. Your own token is not affected, nor is your own assistant in the web UI. What is checked is what is submitted: a run, prediction or optimization file is read once and that reading is checked and submitted; a study file is read once, its cases checked, and every cell checked as the service plans it, with an engine cell's engine (`[plugins] engines`) and each of its predictions as the prediction runs (its simulator and executor, which may replace the run's, must be on the list, and it writes only in the guest's folder). A resume or a prediction checks the configuration saved with the work, so work from before these checks is held to them. A revised study draft's cases must lie in the guest's folders. The editor shows a file only if the file actually opened lies where the guest may read (a link changed after the check is refused).
- The server enforces all of it, not the browser: anything else answers *not found*, and a disallowed action *not allowed*.
- **The agents' buttons**: the run page has *Check health*, *Diagnose* and *Write report*; a finished study's page *Write the study's report*; the editor, on a run configuration, *Plan a study of this case* (a draft in your UI folder). Each runs the agent's rules, or with *ask the model* also the model.
- **The assistant streams** its answer as it is written, with a model that can, and tells you in the chat when a study it proposed and you launched has ended.
- **Separate views**: in the web UI you see only your own work in `UI/owner/`, and each guest only theirs; no one sees the others' jobs, runs, studies, predictions or chats, and your runs elsewhere (the CLI's, older studies) are not shown there. The CLI and the Python client still show you everything.
- **Deleting files**: the editor deletes a file of your own UI folder (`UI/owner/`, a guest's `UI/guests/<name>/`) after you confirm, if it is unchanged since you opened it; emptied folders go too. Files elsewhere are not deleted from the UI.
- **Deleting**: the job, run and study pages have *Delete…*. It shows what would be deleted (jobs, folders, the study's record, the size) and deletes only after you confirm. Each user deletes only their own finished work inside the workspace; a run another submission uses is kept. Folders it empties go too, up to your own UI folder. Guests need `[actions] delete`, which is on by default in a list: an existing `ui_access.toml` without the key lets guests delete their own work; add `delete = false` to prevent it.
- **Where the web UI's work goes**: runs, studies' cells and the assistant's drafts submitted through the web UI are written under the first workspace root's `UI/` folder (`[workspace] ui_folder`, `""` to turn it off): `UI/owner/` for you, `UI/guests/<name>/` for each guest, as `runs/<run>/`, `studies/<study>/...` (studies of optimizations too, whose cells follow the same rules: allowlist, folder, deletion and retention) and `drafts/<chat>/`. A guest may browse their own folder. A reused run (an identical configuration already run) stays where it was first written.
- A guest's configurations must lie in the listed folders and use only the listed plugins.
- Their assistant sees and uses only what they may.
- Without the file, a guest can open and use nothing. Changes apply at once, with no restart.
- The UI shows *guest: NAME*, and hides the buttons for actions that are not allowed.
- **Guests' work is temporary.** Each guest's jobs, runs, studies and predictions are deleted 30 days after submission, and sooner, oldest first, beyond 500 MB per guest. Queued and running jobs are never deleted, nor is your work or any other local program's. Work that cannot be deleted (in use, or a removal that failed) still counts toward the 500 MB, and only what was really removed counts as freed. Folders emptied by a deletion (`runs/`, `studies/`) go too; the guest's own folder stays.
  - Set the limits in `service.toml`: `[guests] retention_days = 30`, `max_storage_mb = 500`.
  - The service applies them hourly; `apex gc --guests` applies them now.

## Pages

| Page | What it shows and does |
|---|---|
| Jobs | The queue by state; a job's status, cells (studies), annotations and live events (SSE, resumed after a dropped connection); *stop* a running run at its next checkpoint, *resume* a stopped or failed one |
| Runs | The catalogue, filtered by annotations (`study=NAME, case=m5spots`); a run's summary, one chart per performance metric over the steps (member mean with the p10–p90 band, the selected step marked; hover for values), the same numbers as a table, and the figures of `apex report --plots` |
| Studies | Studies found through the `study` annotation of runs and of optimizations; a study's progress, its cells (each linked to its run's or its optimization's page) and its comparison (a study of optimizations compares the objective at the end and the best one, both minimised, and the evaluations): each variant's statistics and paired differences against the reference, with Wilcoxon p-values; with methods, a method column and differences against the reference method too |
| Predictions | A prediction's values at the last time per ensemble and vector, and its forecast scores |
| Optimizations | The optimization jobs you may see, newest first, each with its state and its job, whose page stops it while running and resumes it once stopped or failed. An optimization's page shows its optimizer, the objective and which way it goes (every optimizer minimises; a negative `obj_scaling` maximises the quantity, e.g. NPV, and the page shows the unscaled value too, except under EPF, whose penalty is in the value; an older service that sends no scaling shows only "minimised"), whether it is simulated or analytic, the objective over the history (the start, each accepted step and, under EPF, each penalty update, where the line restarts with penalised values), the result or the current point, the objective per realization for a robust optimization, its simulated and failed evaluations, the whole history as a table, and the monitor's and report's buttons. It refreshes while the optimization or its job is queued or running. Submit one from the Editor. Its page also has *Diagnose* (causes and recommended changes) next to *Check health* and *Write report*, marks a result that violates its constraints (constrained runs record `feasible` and `violation`), and deletes a finished optimization (a study's cell: the whole study) after confirming. A multi-objective run (`nsga2`) shows its objectives, its Pareto front (a scatter for two objectives, parallel coordinates for more; the representative ringed, hover a member to read it), the hypervolume over the generations and the front's members with their controls |
| Examples | The catalogue of examples and toy problems (`docs/examples.md`), a tab per workflow: each card says what the example is, where it comes from (e.g. the analytic test problems: pymoo), how long it takes and what it needs. *Copy to my folder* copies it into `UI/<you>/examples/<id>` (a second copy gets `-2`, ...) and opens its configuration in the Editor, where it is validated and submitted like any file of yours. Deck templates stay in the examples root, which the file checks trust. Copying needs `[actions] examples = true` in `ui_access.toml` (off by default). A guest's copies count against `[guests] max_storage_mb` and are deleted after `retention_days`, like their other work |
| Editor | The workspace's `.toml` files: edit with TOML highlighting and a syntax check as you type, *save* (Ctrl+S), *validate* (the errors, or the run plan: members, forecasts, simulator runs, estimated hours) and *submit*; an optimization file (with `[optimization]` and `[optimizer]` tables) shows its plan instead: evaluations, simulator runs and parallel runs, and a study of optimizations (`[study] kind = "optimization"`) its notes (each cell's bound and the study's total). The planner (*Plan a study of this case*) is offered on a saved history-matching or optimization configuration, and the geomodeler (*Give it a prior*) on a saved history-matching one: a new configuration beside it, `<name>_prior.toml`, its prior checked on a drawn sample |
| Plugins | Every plugin, by the workflow it serves: a tab per workflow (history matching, prediction, optimization, simulation & execution, and "Other" for plugins of none), and in it a tab per plugin under its registry, with a search. A plugin's tab shows its status, summary, options (type, default or *required*, allowed values), what the workflow supplies, and its JSON schema; each has an address, `#/plugins/<workflow>/<registry>/<name>`. The service's `workflow` for each plugin decides; for a service that sends none, `ui/src/plugins.ts` maps registries to workflows |
| Agents | APEX's runtime agents (monitor, diagnosis, planner, report, plotter, case-converter, geomodeler): what each does, where to use it in the web UI, its command and its assistant tool. Only `public` agents are shown, here and wherever the UI offers agents. *Bring a case* imports a deck (OPM Flow, Eclipse) with its history as a history-matching case, or creates a synthetic case from a description; either is a draft in `UI/<you>/drafts/case-converter/`, to review and submit in the Editor. Only the owner can also run a created case's truth there (one simulation, for its observations). Give the deck by its workspace path, as the Editor lists files (e.g. `UI/<you>/decks/CASE.DATA`). A deck may include only files its user may read (for a guest: their own folder and the folders `ui_access.toml` lists), and cases are bounded in size for the web UI (`[agents.case_converter] ui_max_cells` and `[agents.geomodeler] ui_max_cells`, 200,000 cells): larger ones are converted on the command line |

## The assistant

*AI Assistant* in the top bar (next to *Agents*) opens a chat panel on every page: 45% of the window wide (480 to 860 px), over the page on windows up to 1100 px, and the whole width on a phone. Esc closes it. It talks to the model of `[agents.llm]` (or `[chat.llm]`) on the service's machine, never from the browser. That model receives your messages and its tools' answers (settings, metric values, well names, paths): a hosted provider on its servers, a local server only on your machines; see [what a provider receives, and how to keep it local](agents.md#what-a-provider-receives-and-how-to-keep-it-local). It can:
- answer from APEX's read-only tools: runs, jobs, studies, predictions and plugins;
- use the agents: `monitor_run`, `diagnose_run`, `write_report` and `list_agents`;
- read a case with `case_info`: its configuration, data, grid and each well's completion cells from the deck (the format of distance localization's `positions`).
- bring a case in: `convert_case` imports a deck with its history as a run draft, `create_case` creates a synthetic case from a description (its truth run is yours to start), and `draft_prior` gives a run draft a prior for its fields.

It knows the page you are on, so "is this run healthy?" on a run's page means that run. It never submits, stops or resumes anything itself.

**Planning a study in the chat.** Ask, for example: *"Plan a comparison of ES-MDA with and without adaptive localization on 5Spots/configs/esmda.toml, within 30,000 simulator runs."*
1. The assistant plans the study (`plan_study`): the planner's code validates and costs every design, and the assistant revises it until it fits the budget. The design it gives is kept (on an optimization case too, where its methods are optimizers), never replaced by the planner's rules. The tool also reports, from the study's cells, the settings all arms share (ensemble size, iterations, simulator, ...), each setting that differs between them, the default values of the options an arm leaves out, and how seeds pair, so the assistant's summary can be checked against them.
2. The draft is written to `UI/<user>/drafts/<chat>/study_<name>.toml` in the workspace (`UI/owner` or `UI/guests/<name>`).
3. A **card** appears: the draft (a link into the editor), its cost, and *Launch*, *Review in the editor* and *Discard*. It appears for every draft the answer wrote, also when the assistant did not offer it itself (then with a generic summary); a stopped answer leaves none. A study whose arms differ in more than the method (budgets, a shared setting, a population, the case's own settings changed) lists that on the card under **Not like for like**. When it compares different optimizers of one family (`ga`, `nsga2`, `nsga3`, `moead`; or `enopt`, `genopt`, `smcopt`), the assistant first gives every arm the first arm's values of the shared operators it leaves unset, and population optimizers the first arm's population, and tells you what it changed; options you set are kept.
4. Review or edit the draft in the editor, or ask the assistant for changes ("three replicates; drop the second variant"). It revises the draft and proposes it again.
5. *Launch* submits the study, as the editor's *Submit* does. Above the service's approval limits, the card shows `apex approve TOKEN` to run on the service's machine, then *Approved: launch*.

**Drafting an optimization in the chat.** Ask, for example: *"Plan an optimization of production by NPV on 5Spots."*
1. The assistant drafts an optimization from the case's finished history-matching run (`draft_optimization`): controls for its wells over a forecast after the history, NPV counted from the end of the history, and robust over the posterior's members. Oil and water prices, the discount rate and (for producers on pressure) their bottom-hole pressures are asked of you; without them it does not draft. The gas price and the objective's scaling, when you don't give them, and the rate bounds (0 to twice the history's largest rate) are assumptions, listed on the card; injectors of other fluids than water keep their last history control.
2. The draft is written to `UI/<user>/drafts/<chat>/` (its configuration, `<name>.toml`, with its deck and model beside it), and a **card** appears: the draft, its cost, the **assumptions** listed in its header (check them before launching), and *Launch*, *Review in the editor* and *Discard*.
3. *Launch* submits the optimization (as the editor does), and it appears on the Optimizations page. A study of optimizations on the draft is planned and launched like any study.

Only your click launches: the assistant has no tool that submits, and its tool session has a budget of 0 simulator runs.

The history is kept on the server, in `<root>/chats.db`:
- **Size limit:** `[chat] max_history_mb`, default 200. Over it, the chats of oldest activity are deleted.
- **Retention:** chats inactive for more than `[chat] retention_days`, default 92 (3 months), are deleted, at service start and daily. `apex gc --chats` applies both limits at once.
- *Delete* removes a chat and the traces of its model calls.

With the `claude_code` provider, each reply takes a few model calls of some seconds each; the panel shows each step while it works. **Stop** (in place of *Send* while it answers) ends the answer at once and frees the chat for a new message. Nothing it did before is undone, such as a draft it wrote. A tool still running is stopped too: the slow read-only ones (`case_info`, `monitor_run`, `diagnose_run`, `write_report`) run in a worker process of their own, which is killed (`[chat] isolate_tools`, on by default). Any other tool is left to finish in the background, and what it gives is dropped. A tool that runs longer than `[chat] tool_timeout_s` (120 s) is abandoned, and the assistant is told so. Without a working model, the chat says why and lists the agents' commands, which also run without a model (`--no-llm`).

### The assistant's models

With `[chat.models]` in `service.toml`, the assistant has a choice of models (otherwise it answers with `[chat.llm]`, else `[agents.llm]`, as above):

```toml
[chat]
default_model = "claude"             # the owner's model when nothing else is picked

[chat.models.claude]                 # each model: the keys of [agents.llm], and a label
label = "Claude"
provider = "claude_code"
model = "sonnet"

[chat.models.beam]
label = "Beam"
provider = "ollama"
base_url = "http://127.0.0.1:11434"
model = "beam:cloud"
```

- **The owner** picks a model at the top of the panel; the next message is answered by it, and the browser remembers the pick for new chats. A chat keeps its model: reopened, it answers with the model of its last answer, and a message sent without a model (over REST) does too. Picking another model in a chat says so above the chat, with a warning when the chat's text would leave the machine (a model on this machine to a hosted one). Each answer names the model that wrote it.
- **Guests** get the models their list names, in order, and do not choose: the panel shows the model and says it is the one available for their account. When a model is unavailable (withdrawn, over its limit, unreachable), the next one answers; with none left, the chat says so. With `[chat.models]` set, their agents' buttons use the same models, never the owner's `[agents.llm]`; the owner's buttons, and other kinds of web-UI user's, keep `[agents.llm]`.

  ```toml
  # ui_access.toml
  [chat]
  models = ["beam", "gemma"]
  ```

  Without `[chat] models`, a guest has no model: the agents' rules answer.
- **Other kinds of web-UI user** (ones an installed package adds) answer with `default_model`.

`GET /v1/chats/models` gives a user's models; a message may name one (`model`), refused (403) when it is not theirs. A model hosted elsewhere (Ollama's cloud models, a hosted API) receives the chat's messages, so choose guests' models with that in mind.

## The editor

- It writes only `.toml` files under the workspace roots (`[workspace] roots` in `service.toml`, default `<root>/configs`). Deck templates and other files cannot be written, since templates are code. Under a `[policy]` the editor is off until roots are set, and they must lie under the policy's `input_roots`.
- Paths are relative to the workspace: with one root, relative to it (`multi_5spots/configs/gies.toml`); with several, prefixed by each root's folder name. The web UI sends them so too (validating, submitting, opening a prediction or optimization), and the server resolves them within the workspace. Run directories, events and errors under the workspace are shown the same way; paths elsewhere (e.g. the service root's `runs/`) are shown in full to you.
- **Guests never see the server's paths**: their workspace lists the roots by name only, and every answer they get names paths under the workspace relative to it, the service's root as `<service>`, and other known folders (frozen code, resources, APEX's source, Python, the temporary folder, the home folder) as placeholders such as `<home>`. A file opened in the editor is shown exactly as it is written, so saving it changes nothing: any absolute paths written in the files of folders listed for guests are therefore shown to them as they are. Only a path outside all of these (another mount named in an error, say) would still appear in full.
- *Validate* checks the text in the editor, saved or not; relative paths resolve against the file's folder.
- *Submit* runs the **saved** file, so save first. A `[study]` file is submitted as a study and a `[prediction]` file as a prediction.
- **Approval.** A submission above the service's approval limits shows its pending token and the command a person runs on the service's machine, `apex approve TOKEN`. Approving is never possible over HTTP. After approval, *submit again with the token*.
- If the file changed on the service while you edited it, saving is refused (`409`) rather than overwriting it; reload the file, or copy your changes out first.

## Development

```bash
npm --prefix ui run dev        # Vite on :5173, proxying /v1 to a service on 127.0.0.1:8765
npm --prefix ui test           # vitest: the API client, SSE parser, routes, chart scales
npm --prefix ui run typecheck
npm --prefix ui run build      # type-checks, then builds ui/dist
```

The code is in `ui/src`: `api.ts` (the client and the documents), `sse.ts`, `router.ts` (hash routes), `chart.ts` (scales, with the palette of `reporting/plots.py`), `pages/` and `components/`.

`npm audit` reports no advisory since vitest 5 and vite 6.4.5; vitest 5 is why the tests need Node 22.12 or later; Node 18.18 still builds it.
