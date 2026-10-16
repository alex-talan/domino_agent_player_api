# source_loader

Resolves local and SharePoint sources into `LoadedSource` objects. Content
extraction remains a separate concern.

## Responsibility

`FileSystemSourceLoader` yields local paths. `SharepointSourceLoader` downloads
remote files as bytes. Both support a single file or a direct folder listing.

## Entities

| Entity | Fields |
|---|---|
| `ContentSource` | `name`, `path`, `type` — describes *what* to load |
| `LoadedSource` | `name`, `content_type`, `content` — one loaded file |

## Port

```python
class SourceLoader(ABC):
    async def load(source: ContentSource) -> AsyncIterator[LoadedSource]: ...
```

## Adapters

| Adapter | Description |
|---|---|
| `FileSystemSourceLoader` | Reads files from the local filesystem. Single file or directory with type filter. |
| `SharepointSourceLoader` | Downloads files through Microsoft Graph. Requires MSAL credentials. |

## Usage

```python
from source_loader import FileSystemSourceLoader, ContentSource

loader = FileSystemSourceLoader()
async for loaded in loader.load(ContentSource(name="docs", path="./sample_docs", type="md")):
    print(loaded.name, loaded.content)
```

LangChain document loaders belong in `content_extraction`: `DirectoryLoader`
returns parsed `Document` objects, while this module resolves sources and
preserves lazy local-file access.
