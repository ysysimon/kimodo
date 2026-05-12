# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for server text encoder startup strategies."""

from __future__ import annotations

import argparse
import os
import sys

import pytest

import kimodo
from kimodo.server.app import create_app
from kimodo.server.runtime import (
    ModelRuntime,
    TextEncoderServerConfig,
    add_text_encoder_args,
    text_encoder_config_from_args,
)


class DummyModel:
    pass


def test_local_text_encoder_mode_sets_local_env(monkeypatch):
    calls = []

    def fake_load_model(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return DummyModel()

    monkeypatch.setattr(kimodo, "load_model", fake_load_model, raising=False)
    runtime = ModelRuntime(
        device="cpu",
        text_encoder_config=TextEncoderServerConfig(mode="local", device="cpu", fp32=True),
    )

    runtime.get_model("kimodo-soma-rp")

    assert os.environ["TEXT_ENCODER_MODE"] == "local"
    assert os.environ["TEXT_ENCODER_DEVICE"] == "cpu"
    assert calls == [
        (
            "kimodo-soma-rp",
            {
                "device": "cpu",
                "default_family": "Kimodo",
                "text_encoder_fp32": True,
            },
        )
    ]


def test_external_text_encoder_mode_sets_api_env_and_probes(monkeypatch):
    probes = []

    monkeypatch.setattr(kimodo, "load_model", lambda *args, **kwargs: DummyModel(), raising=False)
    monkeypatch.setattr(ModelRuntime, "_probe_text_encoder_url", lambda self, url: probes.append(url))
    runtime = ModelRuntime(
        device="cpu",
        text_encoder_config=TextEncoderServerConfig(mode="external", url="http://encoder.example:9550/"),
    )

    runtime.get_model("kimodo-soma-rp")

    assert os.environ["TEXT_ENCODER_MODE"] == "api"
    assert os.environ["TEXT_ENCODER_URL"] == "http://encoder.example:9550/"
    assert probes == ["http://encoder.example:9550/"]


def test_auto_text_encoder_mode_falls_back_to_local(monkeypatch):
    def fail_probe(self, url):
        raise RuntimeError(f"unreachable: {url}")

    monkeypatch.setattr(kimodo, "load_model", lambda *args, **kwargs: DummyModel(), raising=False)
    monkeypatch.setattr(ModelRuntime, "_probe_text_encoder_url", fail_probe)
    runtime = ModelRuntime(
        device="cpu",
        text_encoder_config=TextEncoderServerConfig(mode="auto", url="http://127.0.0.1:9550/"),
    )

    runtime.get_model("kimodo-soma-rp")

    assert os.environ["TEXT_ENCODER_MODE"] == "local"


def test_managed_text_encoder_starts_subprocess_and_closes(monkeypatch):
    commands = []

    class FakeProcess:
        def __init__(self, command, env):
            self.command = command
            self.env = env
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, env):
        process = FakeProcess(command, env)
        commands.append(process)
        return process

    monkeypatch.setattr(kimodo, "load_model", lambda *args, **kwargs: DummyModel(), raising=False)
    monkeypatch.setattr("kimodo.server.runtime.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ModelRuntime, "_probe_text_encoder_url", lambda self, url: None)

    runtime = ModelRuntime(
        device="cpu",
        text_encoder_config=TextEncoderServerConfig(
            mode="managed",
            url="http://127.0.0.1:9660/",
            host="127.0.0.1",
            port=9660,
            tmp_folder="/tmp/kimodo-test-text-encoder/",
            device="cpu",
            fp32=True,
        ),
    )

    runtime.get_model("kimodo-soma-rp")

    process = commands[0]
    assert process.command[:3] == [sys.executable, "-m", "kimodo.scripts.run_text_encoder_server"]
    assert "--fp32" in process.command
    assert process.env["GRADIO_SERVER_NAME"] == "127.0.0.1"
    assert process.env["GRADIO_SERVER_PORT"] == "9660"
    assert process.env["TEXT_ENCODER_DEVICE"] == "cpu"
    assert os.environ["TEXT_ENCODER_MODE"] == "api"
    assert os.environ["TEXT_ENCODER_URL"] == "http://127.0.0.1:9660/"

    runtime.close()

    assert process.terminated is True
    assert process.killed is False


def test_managed_text_encoder_timeout_terminates_subprocess(monkeypatch):
    processes = []

    class FakeProcess:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            raise AssertionError("process should terminate cleanly")

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, env):
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("kimodo.server.runtime.subprocess.Popen", fake_popen)
    runtime = ModelRuntime(
        device="cpu",
        text_encoder_config=TextEncoderServerConfig(mode="managed", startup_timeout_seconds=-1),
    )

    with pytest.raises(RuntimeError, match="Timed out waiting for managed text encoder"):
        runtime.prepare_text_encoder()

    assert processes[0].terminated is True


def test_create_app_accepts_text_encoder_config(tmp_path):
    config = TextEncoderServerConfig(mode="local")

    app = create_app(storage_root=str(tmp_path), text_encoder_config=config)

    assert isinstance(app["runtime"], ModelRuntime)
    assert app["runtime"].text_encoder_config == config
    app["jobs"].executor.shutdown(wait=True)


def test_text_encoder_config_from_cli_args(monkeypatch):
    monkeypatch.setenv("TEXT_ENCODER_MODE", "api")
    parser = add_text_encoder_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--text-encoder-mode",
            "managed",
            "--text-encoder-url",
            "http://127.0.0.1:9770/",
            "--text-encoder-host",
            "0.0.0.0",
            "--text-encoder-port",
            "9770",
            "--text-encoder-fp32",
            "--text-encoder-device",
            "cpu",
        ]
    )

    config = text_encoder_config_from_args(args)

    assert config.mode == "managed"
    assert config.url == "http://127.0.0.1:9770/"
    assert config.host == "0.0.0.0"
    assert config.port == 9770
    assert config.fp32 is True
    assert config.device == "cpu"
