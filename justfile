dev: lint type_check test

alias t := test

@test:
    uv run pytest tests/

alias tc := type_check

@type_check:
    uv run mypy src/ tests/

alias l := lint

@lint:
    uv run ruff format .
    uv run ruff check . --fix