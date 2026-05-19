# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User preferences for the Houdini remote motion client."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_SERVER_URL = f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}"
SERVER_URL_ENV = "KIMODO_REMOTE_MOTION_SERVER_URL"
SERVER_HOST_ENV = "KIMODO_SERVER_HOST"
SERVER_PORT_ENV = "KIMODO_SERVER_PORT"

_CONFIG_DIR_NAME = "kimodo"
_CONFIG_FILE_NAME = "remote_motion.json"
_SERVER_URL_KEY = "server_url"


def get_server_url(*, pref_dir: str | Path | None = None) -> str:
    """Return the configured backend URL for the Houdini client."""
    config = load_config(pref_dir=pref_dir)
    configured_url = config.get(_SERVER_URL_KEY)
    if configured_url:
        return str(configured_url)
    return os.environ.get(SERVER_URL_ENV) or _server_url_from_host_port()


def set_server_url(server_url: str, *, pref_dir: str | Path | None = None) -> Path:
    """Persist the backend URL in the Houdini user preferences directory."""
    normalized_url = normalize_server_url(server_url)
    config = load_config(pref_dir=pref_dir)
    config[_SERVER_URL_KEY] = normalized_url

    path = config_path(pref_dir=pref_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_config(*, pref_dir: str | Path | None = None) -> dict[str, Any]:
    """Load user preferences, returning an empty config when none exists."""
    path = config_path(pref_dir=pref_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def config_path(*, pref_dir: str | Path | None = None) -> Path:
    """Return the per-user Houdini config path used by this integration."""
    root = Path(pref_dir) if pref_dir is not None else _houdini_user_pref_dir()
    return root / _CONFIG_DIR_NAME / _CONFIG_FILE_NAME


def normalize_server_url(server_url: str) -> str:
    """Validate and normalize a backend URL entered by a user."""
    text = server_url.strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Server URL must look like {DEFAULT_SERVER_URL}")
    return text


def _server_url_from_host_port() -> str:
    host = os.environ.get(SERVER_HOST_ENV, DEFAULT_SERVER_HOST)
    port = int(os.environ.get(SERVER_PORT_ENV, DEFAULT_SERVER_PORT))
    return f"http://{host}:{port}"


def _houdini_user_pref_dir() -> Path:
    try:
        import hou  # type: ignore
    except ImportError:
        return Path.home() / ".kimodo" / "houdini"

    pref_dir = hou.expandString("$HOUDINI_USER_PREF_DIR")
    return Path(pref_dir) if pref_dir else Path.home() / ".kimodo" / "houdini"
