# Contributing to clvkit

Thanks for looking. Bug reports and reproducible failures are as useful here as
patches — this library reproduces published estimates, so a case where it does
not is worth knowing about.

## Setup

`clvkit` needs Python 3.13+ and is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nasserboan/clvkit.git
cd clvkit
uv sync
```

Use `uv` for everything. `uv add <pkg>` for a dependency, `uv add --dev <pkg>`
for a dev-only one, `uv run <cmd>` to run anything inside the project
environment. Don't reach for `pip`, `poetry`, or a bare `python`.

## Checks

CI runs these four, and a pull request needs them green:

```bash
uv run ruff check              # lint
uv run ruff format --check     # formatting
uv run pytest                  # 238 tests
uv run pytest --nbmake --nbmake-timeout=900 examples/   # notebooks execute
```

The notebook step is slower than the rest and downloads a dataset on first run.
The unit suite alone is enough while you iterate.

## Tests are the argument

The models implement published papers, and the test suite is what makes that
claim checkable rather than asserted:

- **Golden tests** reproduce the published parameter estimates on the CDNOW
  1/10 sample. If you change a likelihood, these are what catch you.
- **Pinpoint tests** call the private likelihood at fixed parameter points and
  check it against closed forms stated in the paper, or against an independent
  implementation — never against the code path under test.

A change to model maths needs a test that would fail without it. A change that
moves a golden number needs an explanation of why the published value is now
wrong, which is a high bar on purpose.

## Branches and commits

Cut a branch from `main` — `feature/<short-slug>` for changes, `fix/<slug>` for
bugs. `main` takes pull requests only.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `chore:`, and `feat!:` or a `BREAKING CHANGE:` footer
for anything that breaks the public API. The prefix drives the version bump and
the changelog, so a non-conforming message breaks the release.

Keep messages short and direct. The subject carries the change: imperative,
under 72 characters. Add a body only when it says something the diff cannot —
a measured number, an alternative you rejected, a breaking change.

## Pull requests

Pull requests are squash-merged, so the PR title becomes the commit message on
`main`. Write it as the commit you want in the log.

Open a draft early if you want feedback on direction before you finish. Say what
changed and why; if it touches model behaviour, include the numbers before and
after.

## Opinions vs. canon

`opinions.md` records the defaults where the literature genuinely does not
settle the question, separately from what is canon. If your change adds a
default nobody else has to agree with, it belongs there with the alternatives
spelled out. If you think an existing opinion is wrong, that's a conversation
worth opening — bring the case.

## Reporting a bug

Open an issue with the transaction log shape, the calls you made, what you
expected and what you got. A failing snippet against `CDNOW_sample.txt` is the
fastest possible report, since it's in the repo and we can both run it.
