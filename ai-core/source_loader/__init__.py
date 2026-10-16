from importlib import import_module

from domain.entities.content_source import ContentSource
from domain.entities.loaded_source import LoadedSource
from domain.ports.source_loader import SourceLoader

# Loaders load on first access: the SharePoint one pulls msal/requests,
# which need not be installed to read from the filesystem.
_LOADER_MODULES = {
    "FileSystemSourceLoader": "infrastructure.adapters.file_system_source_loader",
    "SharepointSourceLoader": "infrastructure.adapters.sharepoint_source_loader",
}

__all__ = [
    "ContentSource",
    "LoadedSource",
    "SourceLoader",
    *_LOADER_MODULES,
]


def __getattr__(name: str):
    module_path = _LOADER_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
