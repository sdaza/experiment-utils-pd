---
description: Global agent workflow, Python env, and code quality rules
alwaysApply: true
globs: ["*"]
---

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md' with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests → then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to '.cursor/plans/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review to '.cursor/plans/todo.md'
6. **Capture Lessons**: Update '.cursor/lessons.md' after corrections
7. **Update README**: If relevant, update project README with new info or instructions

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Python Environment

### ⚠️ ALWAYS use the project virtual environment
- NEVER use bare `python` or `pip` commands
- ALWAYS use `uv` for package management — NEVER `pip install`
- To install packages: `uv add <package>` or `uv pip install <package>`
- To run Python: `uv run python` or activate venv first with `source .venv/bin/activate`
- To run scripts: `uv run python script.py`
- The venv is at `.venv/` in the project root

## Code Quality

### ⚠️ ALWAYS run ruff before marking any task complete
- ALWAYS run `ruff format .` after making any Python file changes
- ALWAYS run `ruff check .` after making any Python file changes
- Fix ALL ruff errors before considering the task done — never ignore them
- Order: write code → `ruff format .` → `ruff check .` → fix issues → verify again

## RAM Memory usage

- ALWAYS be mindful of RAM usage when running agents, especially with large models or datasets
- When running test check the RAM usage and if it exceeds 80% of available memory, consider optimizing the code or using a smaller model
- Use tools like `htop` or `psutil` to monitor RAM usage in real-time
- If RAM usage is consistently high, investigate memory leaks or inefficient data structures in your code