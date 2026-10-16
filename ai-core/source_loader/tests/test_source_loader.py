import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.entities.content_source import ContentSource
from domain.entities.loaded_source import LoadedSource
from infrastructure.adapters.file_system_source_loader import FileSystemSourceLoader
from infrastructure.adapters.sharepoint_source_loader import (
    SharepointSourceLoader,
    _encode_sharepoint_url,
)


async def _collect_async(async_iterable) -> list:
    items = []
    async for item in async_iterable:
        items.append(item)
    return items


# ── FileSystemSourceLoader ────────────────────────────────────────────────────


class TestFileSystemSourceLoader:
    @pytest.mark.asyncio
    async def test_single_file_yields_one_loaded_source(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello", encoding="utf-8")
        source = ContentSource(name="doc", path=str(f), type="md")

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert len(results) == 1
        assert results[0].name == "doc"
        assert results[0].content == f          # content is a Path
        assert results[0].content_type == "md"

    @pytest.mark.asyncio
    async def test_single_file_infers_type_from_extension_when_type_not_set(
        self, tmp_path: Path
    ):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-fake")
        source = ContentSource(name="report", path=str(f))

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert results[0].content_type == "pdf"

    @pytest.mark.asyncio
    async def test_extensionless_file_requires_a_type(self, tmp_path: Path):
        file_path = tmp_path / "README"
        file_path.write_text("text", encoding="utf-8")
        source = ContentSource(name="readme", path=str(file_path))

        with pytest.raises(ValueError, match="Unable to infer file type"):
            await _collect_async(FileSystemSourceLoader().load(source))

    @pytest.mark.asyncio
    async def test_empty_path_yields_nothing(self):
        source = ContentSource(name="x", path=None)
        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))
        assert results == []

    @pytest.mark.asyncio
    async def test_directory_yields_all_matching_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("A", encoding="utf-8")
        (tmp_path / "b.md").write_text("B", encoding="utf-8")
        (tmp_path / "skip.txt").write_text("skip", encoding="utf-8")
        source = ContentSource(name="docs", path=str(tmp_path), type="md")

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert len(results) == 2
        names = [r.name for r in results]
        assert "a.md" in names
        assert "b.md" in names

    @pytest.mark.asyncio
    async def test_directory_files_are_sorted_alphabetically(self, tmp_path: Path):
        (tmp_path / "z.md").write_text("Z", encoding="utf-8")
        (tmp_path / "a.md").write_text("A", encoding="utf-8")
        (tmp_path / "m.md").write_text("M", encoding="utf-8")
        source = ContentSource(name="docs", path=str(tmp_path), type="md")

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert [r.name for r in results] == ["a.md", "m.md", "z.md"]

    @pytest.mark.asyncio
    async def test_directory_without_type_yields_all_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("A", encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"%PDF")
        source = ContentSource(name="docs", path=str(tmp_path))

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_directory_loaded_source_has_correct_path_and_type(
        self, tmp_path: Path
    ):
        f = tmp_path / "doc.md"
        f.write_text("text", encoding="utf-8")
        source = ContentSource(name="docs", path=str(tmp_path), type="md")

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert results[0].content == f           # content is the Path object
        assert results[0].content_type == "md"

    @pytest.mark.asyncio
    async def test_sub_directories_are_skipped(self, tmp_path: Path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "file.md").write_text("file", encoding="utf-8")
        source = ContentSource(name="docs", path=str(tmp_path), type="md")

        loader = FileSystemSourceLoader()
        results = await _collect_async(loader.load(source))

        assert len(results) == 1
        assert results[0].name == "file.md"


# ── _encode_sharepoint_url ────────────────────────────────────────────────────


class TestEncodeSharepointUrl:
    def test_produces_u_prefix(self):
        result = _encode_sharepoint_url("https://contoso.sharepoint.com/sites/s/file.pdf")
        assert result.startswith("u!")

    def test_no_padding_characters(self):
        result = _encode_sharepoint_url("https://contoso.sharepoint.com/sites/s/file.pdf")
        assert "=" not in result

    def test_roundtrip_via_base64(self):
        url = "https://contoso.sharepoint.com/sites/mysite/Shared%20Documents/report.pdf"
        share_id = _encode_sharepoint_url(url)
        encoded_part = share_id[len("u!"):]
        padded = encoded_part + "=" * (-len(encoded_part) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
        assert decoded == url


# ── SharepointSourceLoader helpers ───────────────────────────────────────────

_SL_MODULE = "source_loader.infrastructure.adapters.sharepoint_source_loader"


def _make_sharepoint_loader(token: str = "fake-token") -> tuple:
    """Return (loader, response_mock) with network I/O mocked out."""
    loader = SharepointSourceLoader(
        tenant_id="tenant-id",
        client_id="client-id",
        client_secret="client-secret",
    )
    msal_mock = MagicMock()
    msal_mock.acquire_token_for_client.return_value = {"access_token": token}
    loader._msal_app = msal_mock

    response_mock = MagicMock()
    response_mock.content = b"%PDF-fake"
    response_mock.raise_for_status.return_value = None
    return loader, response_mock


# ── SharepointSourceLoader unit tests ────────────────────────────────────────


class TestSharepointSourceLoader:
    @pytest.mark.asyncio
    async def test_single_file_yields_one_loaded_source_with_bytes(self):
        loader, response_mock = _make_sharepoint_loader()
        source = ContentSource(
            name="report.pdf",
            path="https://contoso.sharepoint.com/sites/s/report.pdf",
        )

        loader._session = MagicMock()
        loader._session.get.return_value = response_mock
        with patch(f"{_SL_MODULE}.requests.get") as get_mock:
            results = await _collect_async(loader.load(source))

        get_mock.assert_not_called()
        assert len(results) == 1
        assert results[0].name == "report.pdf"
        assert results[0].content_type == "pdf"
        assert results[0].content == b"%PDF-fake"   # content is bytes

    @pytest.mark.asyncio
    async def test_single_file_graph_url_contains_share_id(self):
        loader, response_mock = _make_sharepoint_loader()
        sharepoint_url = "https://contoso.sharepoint.com/sites/mysite/Shared%20Documents/report.pdf"
        source = ContentSource(name="report.pdf", path=sharepoint_url)

        loader._session = MagicMock()
        loader._session.get.return_value = response_mock
        with patch(f"{_SL_MODULE}.requests.get") as get_mock:
            await _collect_async(loader.load(source))

        get_mock.assert_not_called()
        called_url: str = loader._session.get.call_args[0][0]
        expected_share_id = _encode_sharepoint_url(sharepoint_url)
        assert expected_share_id in called_url
        assert called_url.startswith("https://graph.microsoft.com/v1.0/shares/")
        assert called_url.endswith("/content")

    @pytest.mark.asyncio
    async def test_single_file_uses_bearer_token(self):
        loader, response_mock = _make_sharepoint_loader(token="my-secret-token")
        source = ContentSource(
            name="file.pdf",
            path="https://contoso.sharepoint.com/sites/s/file.pdf",
        )

        loader._session = MagicMock()
        loader._session.get.return_value = response_mock
        with patch(f"{_SL_MODULE}.requests.get") as get_mock:
            await _collect_async(loader.load(source))

        get_mock.assert_not_called()
        headers = loader._session.get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-token"

    @pytest.mark.asyncio
    async def test_empty_path_raises_value_error(self):
        loader, _ = _make_sharepoint_loader()
        source = ContentSource(name="x", path=None)
        with pytest.raises(ValueError, match="path is required"):
            await _collect_async(loader.load(source))

    @pytest.mark.asyncio
    async def test_authentication_failure_raises_runtime_error(self):
        loader = SharepointSourceLoader(
            tenant_id="bad-tenant",
            client_id="bad-client",
            client_secret="bad-secret",
        )
        msal_mock = MagicMock()
        msal_mock.acquire_token_for_client.return_value = {
            "error": "invalid_client",
            "error_description": "AADSTS70011: Invalid client credentials.",
        }
        loader._msal_app = msal_mock

        source = ContentSource(
            name="file.pdf",
            path="https://contoso.sharepoint.com/sites/s/file.pdf",
        )
        with pytest.raises(RuntimeError, match="Failed to acquire Microsoft Graph token"):
            await _collect_async(loader.load(source))

    @pytest.mark.asyncio
    async def test_http_error_propagates(self):
        import requests as req

        loader, _ = _make_sharepoint_loader()
        source = ContentSource(
            name="file.pdf",
            path="https://contoso.sharepoint.com/sites/s/file.pdf",
        )
        forbidden = MagicMock()
        forbidden.raise_for_status.side_effect = req.HTTPError("403 Forbidden")

        loader._session = MagicMock()
        loader._session.get.return_value = forbidden
        with patch(f"{_SL_MODULE}.requests.get") as get_mock:
            with pytest.raises(req.HTTPError):
                await _collect_async(loader.load(source))
        get_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_folder_url_lists_children_and_filters_by_type(self):
        loader, _ = _make_sharepoint_loader()
        source = ContentSource(
            name="docs",
            path="https://contoso.sharepoint.com/sites/s/Shared%20Documents",
            type="pdf",
        )
        children_response = MagicMock()
        children_response.raise_for_status.return_value = None
        children_response.json.return_value = {
            "value": [
                {
                    "name": "report.pdf",
                    "file": {},
                    "@microsoft.graph.downloadUrl": "https://cdn.example.com/report.pdf",
                },
                {
                    "name": "notes.txt",
                    "file": {},
                    "@microsoft.graph.downloadUrl": "https://cdn.example.com/notes.txt",
                },
                {
                    "name": "subfolder",
                    # no "file" key → is a folder, should be skipped
                },
            ]
        }
        download_response = MagicMock()
        download_response.content = b"%PDF-fake"
        download_response.raise_for_status.return_value = None

        def _side_effect(url, **kwargs):
            if "children" in url:
                return children_response
            return download_response

        # _list_children and _download_preauthenticated use self._session
        loader._session = MagicMock()
        loader._session.get.side_effect = _side_effect

        results = await _collect_async(loader.load(source))

        assert len(results) == 1
        assert results[0].name == "report.pdf"
        assert results[0].content_type == "pdf"
        assert results[0].content == b"%PDF-fake"

    def test_preauthenticated_download_rejects_private_addresses(self):
        with pytest.raises(ValueError, match="blocked"):
            SharepointSourceLoader._validate_download_url("https://127.0.0.1/file.pdf")

        with pytest.raises(ValueError, match="HTTPS"):
            SharepointSourceLoader._validate_download_url("http://cdn.example.com/file.pdf")
