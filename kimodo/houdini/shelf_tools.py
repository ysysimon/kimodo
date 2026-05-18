# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shelf-tool entry points for Houdini integration."""

from remote_motion_client import RemoteMotionClient


def submit_generation(prompt: str, *, server_url: str = "http://127.0.0.1:8000"):
    """Submit a simple one-prompt generation job from Houdini."""
    client = RemoteMotionClient(server_url)
    return client.submit({"texts": [prompt], "durations": [5.0], "formats": ["bvh"]})
