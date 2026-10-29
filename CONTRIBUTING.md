# Contributing to APEX

Thank you for your interest in APEX. It is in early development, and feedback is very
welcome.

## Issues

Bug reports, questions and feature requests go in the repository's issues. For a bug,
please give the command or configuration you ran, what you expected, what happened (the
full error message), the commit of `main` you used, and your platform. Security issues go
through [SECURITY.md](SECURITY.md) instead, never a public issue.

## Pull requests

This repository is the public release of APEX's development repository: each release is
one commit on top of `main`. Pull requests are welcome, but they are not merged here
directly. An accepted change is carried over into the development repository, with you
credited as its co-author, and appears here in a later release; the pull request is then
closed with a link to that release. A pull request that is not taken is closed with the
reason.

Before you open one:

- keep it small and focused on one change, and say why it is needed;
- run the checks and the tests, as the CI does:

  ```bash
  uv sync --locked --all-packages --extra co2
  uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run lint-imports && uv run pytest -n 4
  uv run mkdocs build --strict    # when you change the documentation
  npm --prefix ui ci && npm --prefix ui run typecheck && npm --prefix ui test    # when you change the web UI (Node 22.12+)
  ```

- add a test that fails without your change, and update the documentation it affects;
- for a new numerical method, cite the paper it implements, and include a test against an
  independent reference (an analytic solution, a published benchmark or a reference
  implementation's stored results).

The architecture (layers that import only inward, plugins chosen by name from the
configuration) is checked by `lint-imports`; a change should extend APEX through its
plugin registries rather than edit existing modules.

## Licence

APEX is licensed under the GNU General Public License version 3 or later. By contributing,
you agree that your contribution is licensed under the same terms.
