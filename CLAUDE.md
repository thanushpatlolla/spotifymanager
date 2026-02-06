# CLAUDE.md - Instructions for Claude Code

## Project
Spotify playlist manager — CLI tool using Spotify + Last.fm APIs, SQLite, SQLAlchemy 2.0.

## Git & Commit Policy
- **Commit regularly**: After completing any significant change (new feature, bug fix, refactor, or milestone), create a git commit before moving on.
- **Push after commits**: Push to `origin main` after committing so work is preserved remotely.
- **Commit message style**: Short imperative subject line (e.g. "Add listening stats CLI command"), optional body for context.
- **What counts as "significant"**: Any completed feature, fixed bug, new module, test addition, config change, or structural refactor. When in doubt, commit.
- **Do NOT commit**: Half-finished work, broken code, or secrets (.env files). Run tests before committing when possible.

## Build & Run
- Package manager: `uv`
- Install: `uv pip install -e ".[dev]"`
- Run CLI: `uv run spotifymanager`
- Run tests: `uv run pytest`

## Code Conventions
- SQLAlchemy 2.0 modern style (Mapped[T], mapped_column)
- Click for CLI
- Config in `config.yaml`, secrets in `.env`
