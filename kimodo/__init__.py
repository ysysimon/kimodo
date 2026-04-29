# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo: text-driven and constrained motion generation model."""

from .model.registry import AVAILABLE_MODELS, DEFAULT_MODEL

__all__ = [
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "load_model",
]


def __getattr__(name: str):
    if name == "load_model":
        from .model.load_model import load_model

        globals()[name] = load_model
        return load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
