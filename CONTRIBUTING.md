# Contributing to GuidedVLA

We welcome contributions, bug reports, feature requests, and documentation
improvements. GuidedVLA is released under the repository license, with
additional third-party terms noted in the README and in the relevant
submodules.

## Issues and feature requests

Use GitHub issues for bugs and feature requests:

- Issues: https://github.com/GuidedVLA/GuidedVLA/issues
- Discussions: https://github.com/GuidedVLA/GuidedVLA/discussions

For bugs, include:

- Your OS, Python version, CUDA/GPU details when relevant, and install command
- The exact command or code needed to reproduce the issue
- Full traceback or error output
- Any relevant config name, checkpoint path, dataset format, or submodule state

For feature requests, include the motivation, the intended workflow, and enough
context for maintainers to evaluate the implementation and maintenance cost.

## Pull requests

Before opening a pull request:

- Make sure the PR has a clear title and description.
- Install hooks with `pre-commit install`.
- Run `pre-commit run --all-files`, or at minimum `ruff check .`,
  `ruff format .`, and the relevant tests.
- Keep changes scoped. Separate mechanical cleanup, behavior changes, and large
  refactors into different PRs when possible.

For model, data-loading, or training changes, include the config and command you
used for validation. For user-facing workflows, update the relevant README or
docs page in the same PR.
