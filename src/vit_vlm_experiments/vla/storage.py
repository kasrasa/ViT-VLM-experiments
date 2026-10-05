from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file only after its complete replacement is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
