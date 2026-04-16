from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path) -> Path:
    """Walk upward from *start* until a directory containing both
    pyproject.toml and configs/demo.yaml is found.

    This allows generated scripts stored deep inside logs/.../attempts/
    to still locate the true repository root.
    """
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent

    for candidate in [path, *path.parents]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'configs' / 'demo.yaml').exists():
            return candidate

    raise FileNotFoundError(
        f'Could not locate project root from {start!s}; expected a parent directory containing '
        'pyproject.toml and configs/demo.yaml.'
    )


def resolve_project_path(project_root: str | Path, maybe_relative: str | Path) -> Path:
    """Return an absolute path under project_root unless maybe_relative is already absolute."""
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(project_root).resolve() / path
