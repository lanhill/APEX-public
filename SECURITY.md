# Security policy

## Supported versions

Only the latest commit of `main` is supported. APEX is in early development: fixes are made
on `main` only and reach users in the next release, not as patches to earlier tagged releases
(the first is v0.1.0).

## Reporting a vulnerability

Please do not report security issues in public issues, pull requests or discussions.

Report them privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. Please include:

- what is affected (a command, a REST route, the web UI, an agent tool, a configuration
  option) and the commit you used;
- the steps to reproduce it, and what an attacker could do with it;
- any fix or mitigation you suggest.

We will acknowledge your report as soon as we can, keep you informed while we work on it,
and credit you in the fix unless you prefer otherwise.

## Scope notes

- The REST API listens on the address given to `apex service start --http`, and requires
  a bearer token. Keep it on `127.0.0.1`, as the documentation does; to reach it from
  other machines, put it behind a reverse proxy with TLS, and keep the tokens secret.
- Configurations can name simulators, programs and files. Run configurations only from
  sources you trust, or with a `[policy]` that confines what they may name (see
  [docs/service.md](docs/service.md#policy) and [docs/tools.md](docs/tools.md)).
- Runtime agents send the facts of a run to the language model you configure. Choose a
  provider whose data handling suits your data.
