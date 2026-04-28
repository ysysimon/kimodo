# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shelf-tool entry points for Houdini integration."""

from .client import KimodoClient


def submit_generation(prompt: str, *, server_url: str = "http://127.0.0.1:8765"):
    """Submit a simple one-prompt generation job from Houdini."""
    client = KimodoClient(server_url)
    return client.submit(texts=[prompt], durations=[5.0], formats=["npz"])
