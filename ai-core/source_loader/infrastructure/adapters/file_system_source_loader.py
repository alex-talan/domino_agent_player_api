from pathlib import Path
from typing import AsyncIterator

from domain.entities.content_source import ContentSource
from domain.entities.loaded_source import LoadedSource
from domain.ports.source_loader import SourceLoader


class FileSystemSourceLoader(SourceLoader[Path]):
    """Resolve a local file or its immediate directory entries to paths."""

    async def load(self, source: ContentSource) -> AsyncIterator[LoadedSource[Path]]:
        if not source.path:
            return

        path = Path(source.path)

        try:
            if path.is_file():
                ext = path.suffix.lstrip(".").lower()
                content_type = source.type or ext
                if not content_type:
                    raise ValueError(f"Unable to infer file type from: {path}")
                yield LoadedSource(
                    name=source.name or path.name,
                    content_type=content_type,
                    content=path,
                )

            elif path.is_dir():
                ext = (source.type or "").lower().lstrip(".")
                for file_path in sorted(path.iterdir(), key=lambda item: item.name):
                    if not file_path.is_file():
                        continue
                    file_ext = file_path.suffix.lstrip(".").lower()
                    if ext and file_ext != ext:
                        continue
                    if not file_ext:
                        continue
                    yield LoadedSource(
                        name=file_path.name,
                        content_type=file_ext,
                        content=file_path,
                    )

            else:
                raise FileNotFoundError(f"Source path does not exist: {path}")
            
        except OSError as exc:
            raise OSError(f"Unable to access filesystem source: {path}") from exc