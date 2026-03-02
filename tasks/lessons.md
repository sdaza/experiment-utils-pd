# Lessons

## After any code change, reinstall the package
Run `uv pip install -e .` after modifying source files so the installed
version stays in sync. Without this, users running from `site-packages`
hit stale code and confusing errors. Always instruct users to restart
their Jupyter kernel after reinstalling.

## Always run tests before declaring a task done
Never mark a task complete after just linting. Run `uv run pytest` (or the
relevant test subset) and confirm all tests pass. If no tests exist for the
changed code, write them first — then run them. A green ruff check is not
proof the code works.
