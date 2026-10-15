import os
from pathlib import Path

from dotenv import dotenv_values

_DEFAULT_ENV_FILES = (".env", ".env.local")
_ENV_FILE_VAR = "WIP_ANALYSIS_ENV_FILE"
_LOADED = False
_LOADED_KEYS: set[str] = set()


def _resolve_path(path: str, cwd: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return cwd / expanded


def _find_default_root(cwd: Path) -> Path:
    for root in (cwd, *cwd.parents):
        if any((root / name).is_file() for name in _DEFAULT_ENV_FILES):
            return root
        if (root / "pyproject.toml").is_file() and (root / "src").is_dir():
            return root
    return cwd


def _env_file_paths(cwd: Path) -> list[Path]:
    explicit = os.environ.get(_ENV_FILE_VAR)
    if explicit:
        return [
            _resolve_path(part.strip(), cwd) for part in explicit.split(os.pathsep) if part.strip()
        ]

    root = _find_default_root(cwd)
    return [root / name for name in _DEFAULT_ENV_FILES]


def load_runtime_env(*, cwd: str | Path | None = None, force: bool = False) -> tuple[Path, ...]:
    """Load project runtime variables from dotenv files without overriding process env.

    Process-provided variables always win. When both default files exist, values in
    .env.local override .env values that were not already present in the process.
    """
    global _LOADED

    if _LOADED and not force:
        return ()

    cwd_path = Path.cwd() if cwd is None else Path(cwd)
    protected_env = set(os.environ) - _LOADED_KEYS
    loaded: list[Path] = []

    for env_file in _env_file_paths(cwd_path):
        if not env_file.is_file():
            continue

        for key, value in dotenv_values(env_file).items():
            if value is None or key in protected_env:
                continue
            os.environ[key] = value
            _LOADED_KEYS.add(key)
        loaded.append(env_file)

    _LOADED = True
    return tuple(loaded)
