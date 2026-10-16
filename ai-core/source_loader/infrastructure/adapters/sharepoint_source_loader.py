"""SharePoint file resolution through Microsoft Graph and MSAL."""
from __future__ import annotations

import asyncio
import base64
import ipaddress
from pathlib import PurePosixPath
from typing import AsyncIterator
from urllib.parse import parse_qs, unquote, urlparse

import msal
import requests

from domain.entities.content_source import ContentSource
from domain.entities.loaded_source import LoadedSource
from domain.ports.source_loader import SourceLoader

_GRAPH_SCOPE = "https://graph.microsoft.com/.default"
_GRAPH_DRIVES_BASE = "https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem"


def _encode_sharepoint_url(url: str) -> str:
    """Encode a SharePoint URL into a Graph API share identifier.

    Steps: base64url-encode the URL → strip ``=`` padding → prefix ``u!``.
    See https://learn.microsoft.com/en-us/graph/api/shares-get
    """
    normalized_url = url.strip()
    parsed = urlparse(normalized_url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid SharePoint URL: {url!r}")

    encoded = base64.urlsafe_b64encode(normalized_url.encode("utf-8")).decode("ascii").rstrip("=")

    return f"u!{encoded}"


class SharepointSourceLoader(SourceLoader[bytes]):
    """Download files from SharePoint through Microsoft Graph.
    Yields ``LoadedSource[bytes]`` objects; ``content`` holds the raw file
    bytes downloaded from SharePoint.

    Parameters
    ----------
    tenant_id:
        Azure / Entra ID tenant identifier (GUID or domain).
    client_id:
        Application (client) ID of the registered Entra ID app.
    client_secret:
        Client secret of the registered Entra ID app.
    request_timeout:
        HTTP timeout in seconds for Graph API calls.  Default 60.
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        request_timeout: float = 60,
    ) -> None:
        tenant_id = tenant_id.strip()
        client_id = client_id.strip()

        if not tenant_id:
            raise ValueError("tenant_id must not be empty.")
        if not client_id:
            raise ValueError("client_id must not be empty.")
        if not client_secret:
            raise ValueError("client_secret must not be empty.")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero.")

        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._request_timeout = request_timeout
        self._msal_app: msal.ConfidentialClientApplication | None = None
        self._session = requests.Session()

    # ------------------------------------------------------------------ auth

    def _get_msal_app(self) -> msal.ConfidentialClientApplication:
        if self._msal_app is None:
            self._msal_app = msal.ConfidentialClientApplication(
                client_id=self._client_id,
                client_credential=self._client_secret,
                authority=f"https://login.microsoftonline.com/{self._tenant_id}",
            )
        return self._msal_app

    def _acquire_token(self) -> str:
        """Return a cached-or-fresh access token for the Microsoft Graph API.

        Raises
        ------
        RuntimeError
            If MSAL cannot obtain a token (wrong credentials, tenant, or
            missing application permissions).
        """
        app = self._get_msal_app()
        result = app.acquire_token_for_client(scopes=[_GRAPH_SCOPE])
        if not isinstance(result, dict):
            raise RuntimeError("Failed to acquire Microsoft Graph token: invalid MSAL response")
        if not isinstance(result.get("access_token"), str):
            description = result.get("error_description") or result.get("error", "unknown")
            raise RuntimeError(f"Failed to acquire Microsoft Graph token: {description}")
        return result["access_token"]

    # --------------------------------------------------------------- download

    def _download_with_token(self, url: str, token: str) -> bytes:
        """Download bytes from a URL using a Bearer token."""
        response = self._session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=True,
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        return response.content

    def _download_file(self, sharepoint_url: str, token: str) -> bytes:
        """Download a single SharePoint file via the sharing-link endpoint."""
        share_id = _encode_sharepoint_url(sharepoint_url)
        url = _GRAPH_DRIVES_BASE.format(share_id=share_id) + "/content"
        return self._download_with_token(url, token)

    def _list_children(self, sharepoint_url: str, token: str) -> list[dict]:
        share_id = _encode_sharepoint_url(sharepoint_url)
        next_url: str | None = (
            _GRAPH_DRIVES_BASE.format(share_id=share_id) + "/children"
        )
        children: list[dict] = []

        headers = {"Authorization": f"Bearer {token}"}

        while next_url:
            try:
                response = self._session.get(
                    next_url,
                    headers=headers,
                    timeout=self._request_timeout,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    "Failed to list children of the SharePoint folder."
                ) from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    "Microsoft Graph returned invalid JSON for folder contents."
                ) from exc
            values = payload.get("value", [])

            if not isinstance(values, list):
                raise RuntimeError(
                    "Microsoft Graph returned an invalid children response."
                )

            children.extend(
                item for item in values if isinstance(item, dict)
            )

            next_link = payload.get("@odata.nextLink")
            if next_link is not None and not isinstance(next_link, str):
                raise RuntimeError("Microsoft Graph returned an invalid next link.")
            if isinstance(next_link, str):
                parsed_next = urlparse(next_link)
                if parsed_next.scheme != "https" or parsed_next.hostname != "graph.microsoft.com":
                    raise RuntimeError("Microsoft Graph returned an unsafe next link.")
            next_url = next_link

        return children

    def _download_preauthenticated(self, url: str) -> bytes:
        self._validate_download_url(url)
        try:
            response = self._session.get(
                url,
                allow_redirects=True,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            raise RuntimeError(
                "Failed to download a SharePoint file."
            ) from exc

    @staticmethod
    def _validate_download_url(url: str) -> None:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
            raise ValueError("SharePoint download URLs must use HTTPS without credentials.")

        normalized_host = hostname.lower().rstrip(".")
        if normalized_host in {"localhost", "metadata.google.internal"}:
            raise ValueError("SharePoint download URL points to a blocked host.")
        try:
            address = ipaddress.ip_address(normalized_host)
        except ValueError:
            return
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise ValueError("SharePoint download URL points to a blocked address.")

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _is_file_url(path: str) -> bool:
        parsed = urlparse(path)
        query = parse_qs(parsed.query)

        # SharePoint list/library views such as Forms/AllItems.aspx are not files.
        last_segment = PurePosixPath(unquote(parsed.path)).name
        if last_segment.lower() in {
            "allitems.aspx",
            "forms.aspx",
            "dispform.aspx",
        }:
            return False

        # Some SharePoint URLs carry the actual target in the `id` parameter.
        target_values = query.get("id")
        target_path = (
            unquote(target_values[0])
            if target_values
            else unquote(parsed.path)
        )

        target_name = PurePosixPath(target_path.rstrip("/")).name
        suffix = PurePosixPath(target_name).suffix

        return bool(suffix)

    # ------------------------------------------------------------------ port

    async def load(self, source: ContentSource) -> AsyncIterator[LoadedSource[bytes]]:
        if not source.path:
            raise ValueError("ContentSource.path is required.")

        token = await asyncio.to_thread(self._acquire_token)

        if self._is_file_url(source.path):
            # ── single file ───────────────────────────────────────────────
            name = PurePosixPath(source.path.split("?")[0]).name
            content_bytes = await asyncio.to_thread(
                self._download_file, source.path, token
            )
            ext = PurePosixPath(name).suffix.lstrip(".").lower()
            yield LoadedSource(
                name=source.name or name,
                content_type=ext,
                content=content_bytes,
            )

        else:
            # ── folder: list children and filter by type ──────────────────
            target_ext = (source.type or "").lower().lstrip(".")
            children = await asyncio.to_thread(
                self._list_children, source.path, token
            )
            for item in sorted(children, key=lambda child: str(child.get("name", ""))):
                if "file" not in item:
                    continue  # skip sub-folders
                item_name = item.get("name", "")
                item_ext = PurePosixPath(item_name).suffix.lstrip(".").lower()
                if target_ext and item_ext != target_ext:
                    continue
                download_url = item.get("@microsoft.graph.downloadUrl")
                if not download_url:
                    continue
                content_bytes = await asyncio.to_thread(
                    self._download_preauthenticated, download_url
                )
                yield LoadedSource(
                    name=item_name,
                    content_type=item_ext,
                    content=content_bytes,
                )
