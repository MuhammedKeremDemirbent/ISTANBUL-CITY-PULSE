# Local Task Plan: Translate Comments, Docstrings, and Logs to English

## Plan / Requirements
- [ ] Scan project for Turkish text across `src/`, `dags/`, `tests/`, `Makefile`, etc. (excluding `ZZ_NOTLAR/`, `.venv/`, `.git/`, `data/`).
- [ ] Translate all comments (`# ...`) to concise, simple English.
- [ ] Translate all docstrings (`"""..."""`) to clean, simple English.
- [ ] Translate all log messages (`log.info`, `log.warning`, `log.error`, `log.debug`) to simple English.
- [ ] Translate print statements and console messages (e.g. in `sql_shell.py`, `dashboard/app.py`, etc.) to simple English.
- [ ] Ensure no tests break (especially if any test checks exact log/message strings).
- [ ] Run full verification:
  - [ ] `uv run pytest`
  - [ ] `uv run ruff check .`
  - [ ] `uv run ruff format --check .`
  - [ ] `uv run mypy src`
- [ ] Archive plan to `docs/archived/plans/local-english-comments-logs/`.

## Expected Verification
- `uv run pytest` -> 41 passed
- `uv run ruff check .` -> clean
- `uv run ruff format --check .` -> clean
- `uv run mypy src` -> clean

## Progress
- [ ] Task plan created.
- [ ] Turkish text scan and file list identified.
- [ ] Updates applied to `src/city_pulse/`.
- [ ] Updates applied to `dags/`.
- [ ] Updates applied to `tests/`.
- [ ] Updates applied to other root files (`Makefile`, etc.).
- [ ] Verification complete.
- [ ] Plan archived.

## Blockers / Open Questions
- None.
