# Repository Guidelines

## Project Structure & Module Organization
This repository is currently lightweight, so keep the top level tidy and predictable. Place application code in `src/`, tests in `tests/`, and supporting assets in `assets/` or `docs/` as the project grows. Keep one responsibility per module and group related files by feature rather than by file type when that becomes practical.

Examples:
- `src/<feature>/...`
- `tests/test_<feature>.*`
- `docs/architecture.md`

## Build, Test, and Development Commands
No single build system is established in the repository yet, so contributors should prefer explicit, scriptable commands and document any new tooling they add. If you introduce a runtime or package manager, add the common entry points to the repository root and update this guide in the same change.

Typical command patterns:
- `npm test` or `pytest`: run the automated test suite.
- `npm run build` or `python -m build`: create a production build.
- `npm run dev` or similar: start a local development workflow.

## Coding Style & Naming Conventions
Use consistent formatting and keep files small enough to review quickly. Prefer 4 spaces for Python, 2 spaces for JSON/YAML/Markdown indentation where applicable, and descriptive names over abbreviations.

Naming guidance:
- Files/modules: `snake_case` for Python, `kebab-case` for docs/assets.
- Classes/types: `PascalCase`.
- Functions/variables: `snake_case` or the language’s standard local convention.

If you add a formatter or linter, commit its config with the change.

## Testing Guidelines
Add tests alongside every non-trivial behavior change. Use clear, behavior-based test names such as `test_rejects_invalid_input` or `should_render_empty_state`. Keep unit tests fast and isolate external services behind mocks or fixtures.

Aim to cover:
- Happy-path behavior
- Edge cases and invalid input
- Regressions for reported bugs

## Commit & Pull Request Guidelines
Keep commits focused and readable. Use short, imperative commit subjects such as `Add input validation` or `Document local setup`. Avoid mixing refactors with behavior changes unless the refactor is required for the fix.

Pull requests should include:
- A concise description of the change
- Test evidence or a note explaining why tests were not added
- Linked issue or task, if one exists
- Screenshots only when UI output changed

## Agent-Specific Instructions
When editing this repository, prefer small patches, avoid unrelated churn, and update `AGENTS.md` whenever new tooling, directories, or review expectations become standard.
