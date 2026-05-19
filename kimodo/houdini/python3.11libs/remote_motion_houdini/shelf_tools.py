# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shelf-tool entry points for Houdini integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from remote_motion_client import RemoteMotionClient

from .cache import default_download_dir
from .config import get_server_url, normalize_server_url, set_server_url

_FILETYPE_PARM = "filetype"
_FILETYPE_BIOVISION_VALUE = "biovision"
_BVH_FILE_PARM = "bvhfile"
_DEFAULT_DURATION = 5.0
_SERVER_DEFAULT_MODEL_LABEL = "Server default"
_RELOAD_PARMS = ("reload", "reloadfile")
_MOCAP_IMPORT_NODE_TYPES = ("mocapimport", "mocapimport::2.0", "kinefx::mocapimport", "kinefx::mocapimport::2.0")


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    duration: float = _DEFAULT_DURATION
    seed: int | None = None
    model: str | None = None


def submit_generation(
    prompt: str,
    *,
    duration: float = _DEFAULT_DURATION,
    seed: int | None = None,
    model: str | None = None,
    server_url: str | None = None,
):
    """Submit a simple one-prompt generation job from Houdini."""
    client = RemoteMotionClient(server_url or get_server_url())
    return client.submit(_generation_payload(prompt, duration=duration, seed=seed, model=model))


def generate_and_download(
    prompt: str,
    *,
    duration: float = _DEFAULT_DURATION,
    seed: int | None = None,
    model: str | None = None,
    server_url: str | None = None,
    poll_interval: float = 1.0,
    wait_timeout: float | None = None,
) -> Path:
    """Generate one BVH motion and download the artifact."""
    client = RemoteMotionClient(server_url or get_server_url())
    submitted = client.submit(_generation_payload(prompt, duration=duration, seed=seed, model=model))
    job = client.wait(submitted.job_id, poll_interval=poll_interval, timeout=wait_timeout)
    return client.download_artifact(job, "bvh", default_download_dir(job.job_id))


def create_mocap_import_node(artifact_path: str | Path, prompt: str | None = None):
    """Create a standalone Houdini Mocap Import node for a downloaded BVH."""
    import hou  # type: ignore

    obj = hou.node("/obj")
    if obj is None:
        raise RuntimeError("Could not find Houdini /obj network.")

    geo_name_source = prompt or Path(artifact_path).stem
    geo = _create_named_node(obj, "geo", _safe_node_name("kimodo_motion", geo_name_source))
    _remove_default_file_node(geo)
    mocap = _create_first_available_node(geo, _MOCAP_IMPORT_NODE_TYPES, "mocap_import")
    _set_required_parm(mocap, _FILETYPE_PARM, _FILETYPE_BIOVISION_VALUE)
    _set_required_parm(mocap, _BVH_FILE_PARM, _as_hip_relative_path(artifact_path, hou))
    _press_first_existing_button(mocap, _RELOAD_PARMS)

    _set_display_flags(mocap)
    geo.layoutChildren()
    obj.layoutChildren()
    try:
        mocap.setCurrent(True, clear_all_selected=True)
    except TypeError:
        mocap.setCurrent(True)

    return mocap


def prompt_submit_generation() -> None:
    """Prompt for text, generate a motion, and download the BVH artifact."""
    import hou  # type: ignore

    settings = _prompt_generation_settings(hou)
    if settings is None:
        return

    artifact_path = generate_and_download(
        settings.prompt,
        duration=settings.duration,
        seed=settings.seed,
        model=settings.model,
    )
    mocap = create_mocap_import_node(artifact_path, settings.prompt)
    hou.ui.displayMessage(f"Kimodo motion downloaded:\n{artifact_path}\n\nCreated node:\n{mocap.path()}")


def configure_server() -> None:
    """Prompt for and persist the backend URL used by Houdini client tools."""
    import hou  # type: ignore

    button, server_url = hou.ui.readInput(
        "Kimodo backend URL",
        buttons=("Save", "Cancel"),
        initial_contents=get_server_url(),
    )
    if button != 0:
        return

    try:
        normalized_url = normalize_server_url(server_url)
        path = set_server_url(normalized_url)
    except ValueError as exc:
        hou.ui.displayMessage(str(exc), severity=hou.severityType.Error)
        return

    hou.ui.displayMessage(f"Kimodo backend URL saved:\n{normalized_url}\n\n{path}")


def _generation_payload(prompt: str, *, duration: float, seed: int | None, model: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "texts": [prompt],
        "durations": [duration],
        "formats": ["bvh"],
    }
    if seed is not None:
        payload["seed"] = seed
    if model:
        payload["model"] = model
    return payload


def _prompt_generation_settings(hou_module) -> GenerationSettings | None:
    from hutil.Qt import QtCore, QtWidgets  # type: ignore

    dialog = _GenerationSettingsDialog(
        QtCore,
        QtWidgets,
        model_names=_available_model_names(),
        parent=_main_qt_window(hou_module),
    )
    exec_dialog = getattr(dialog, "exec_", None) or getattr(dialog, "exec")
    if exec_dialog() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.settings()


def _main_qt_window(hou_module):
    qt = getattr(hou_module, "qt", None)
    if qt is None:
        return None

    main_window = getattr(qt, "mainWindow", None)
    if main_window is None:
        return None

    try:
        return main_window()
    except Exception:
        return None


class _GenerationSettingsDialog:
    def __new__(cls, qt_core, qt_widgets, *, model_names: list[str], parent=None):
        class Dialog(qt_widgets.QDialog):
            def __init__(self) -> None:
                super().__init__(parent)
                self.setWindowTitle("Kimodo Motion Generation")
                self.setMinimumWidth(420)

                self.prompt_edit = qt_widgets.QLineEdit(self)
                self.prompt_edit.setPlaceholderText("walk forward")

                self.model_combo = qt_widgets.QComboBox(self)
                self.model_combo.addItem(_SERVER_DEFAULT_MODEL_LABEL, None)
                for model_name in model_names:
                    self.model_combo.addItem(model_name, model_name)

                horizontal = getattr(getattr(qt_core.Qt, "Orientation", qt_core.Qt), "Horizontal")

                self.duration_slider = qt_widgets.QSlider(horizontal, self)
                self.duration_slider.setRange(1, 600)
                self.duration_slider.setValue(int(_DEFAULT_DURATION * 10))

                self.duration_spin = qt_widgets.QDoubleSpinBox(self)
                self.duration_spin.setRange(0.1, 60.0)
                self.duration_spin.setDecimals(1)
                self.duration_spin.setSingleStep(0.1)
                self.duration_spin.setSuffix(" s")
                self.duration_spin.setValue(_DEFAULT_DURATION)

                self.seed_enabled = qt_widgets.QCheckBox("Use fixed seed", self)
                self.seed_slider = qt_widgets.QSlider(horizontal, self)
                self.seed_slider.setRange(0, 999999)
                self.seed_spin = qt_widgets.QSpinBox(self)
                self.seed_spin.setRange(0, 999999)

                self.seed_slider.setEnabled(False)
                self.seed_spin.setEnabled(False)

                self.buttons = qt_widgets.QDialogButtonBox(
                    qt_widgets.QDialogButtonBox.Ok | qt_widgets.QDialogButtonBox.Cancel,
                    self,
                )

                form = qt_widgets.QFormLayout()
                form.addRow("Prompt", self.prompt_edit)
                form.addRow("Model", self.model_combo)
                form.addRow("Duration", _row(qt_widgets, self.duration_slider, self.duration_spin))
                form.addRow("", self.seed_enabled)
                form.addRow("Seed", _row(qt_widgets, self.seed_slider, self.seed_spin))

                layout = qt_widgets.QVBoxLayout(self)
                layout.addLayout(form)
                layout.addWidget(self.buttons)

                self.duration_slider.valueChanged.connect(lambda value: self.duration_spin.setValue(value / 10.0))
                self.duration_spin.valueChanged.connect(lambda value: self.duration_slider.setValue(round(value * 10)))
                self.seed_enabled.toggled.connect(self.seed_slider.setEnabled)
                self.seed_enabled.toggled.connect(self.seed_spin.setEnabled)
                self.seed_slider.valueChanged.connect(self.seed_spin.setValue)
                self.seed_spin.valueChanged.connect(self.seed_slider.setValue)
                self.buttons.accepted.connect(self.accept)
                self.buttons.rejected.connect(self.reject)

            def accept(self) -> None:
                if not self.prompt_edit.text().strip():
                    qt_widgets.QMessageBox.warning(self, "Kimodo Motion Generation", "Prompt cannot be empty.")
                    return
                super().accept()

            def settings(self) -> GenerationSettings:
                model = self.model_combo.currentData()
                seed = self.seed_spin.value() if self.seed_enabled.isChecked() else None
                return GenerationSettings(
                    prompt=self.prompt_edit.text().strip(),
                    duration=float(self.duration_spin.value()),
                    seed=seed,
                    model=model,
                )

        return Dialog()


def _row(qt_widgets, *widgets):
    row = qt_widgets.QWidget()
    layout = qt_widgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return row


def _available_model_names() -> list[str]:
    try:
        from kimodo.model.registry import AVAILABLE_MODELS, DEFAULT_MODEL
    except Exception:
        return []

    names = list(dict.fromkeys(AVAILABLE_MODELS))
    if DEFAULT_MODEL in names:
        names.remove(DEFAULT_MODEL)
        names.insert(0, DEFAULT_MODEL)
    return names


def _create_first_available_node(parent, node_types: tuple[str, ...], node_name: str):
    errors = []
    for node_type in node_types:
        try:
            return _create_named_node(parent, node_type, node_name)
        except Exception as exc:  # Houdini raises hou.OperationFailed for unknown node types.
            errors.append(f"{node_type}: {exc}")
    raise RuntimeError("Could not create a Mocap Import node. Tried: " + "; ".join(errors))


def _create_named_node(parent, node_type: str, node_name: str):
    node = parent.createNode(node_type)
    set_name = getattr(node, "setName", None)
    if set_name is not None:
        set_name(node_name, unique_name=True)
    return node


def _set_required_parm(node, parm_name: str, value: str) -> None:
    parm = node.parm(parm_name)
    if parm is None:
        raise RuntimeError(
            f"Could not find required parameter '{parm_name}' on {node.path()}. "
            f"Available parms: {_parm_names_text(node)}"
        )
    parm.set(value)


def _as_hip_relative_path(path: str | Path, hou_module) -> str:
    artifact = Path(path)
    try:
        hip = hou_module.expandString("$HIP")
    except Exception:
        hip = ""

    if not hip:
        return str(artifact)

    try:
        relative_path = artifact.resolve().relative_to(Path(hip).resolve())
    except ValueError:
        return str(artifact)

    return "$HIP/" + relative_path.as_posix()


def _parm_text(method) -> str:
    if method is None:
        return ""
    try:
        value = method()
    except Exception:
        return ""
    return str(value)


def _parm_names_text(node) -> str:
    parms = getattr(node, "parms", None)
    if not callable(parms):
        return "<not available>"

    try:
        names = [_parm_text(getattr(parm, "name", None)) for parm in parms()]
    except Exception:
        return "<not available>"

    names = [name for name in names if name]
    return ", ".join(names) if names else "<none>"


def _press_first_existing_button(node, parm_names: tuple[str, ...]) -> None:
    for parm_name in parm_names:
        parm = node.parm(parm_name)
        if parm is not None:
            parm.pressButton()
            return


def _remove_default_file_node(geo) -> None:
    file_node = geo.node("file1")
    if file_node is not None:
        file_node.destroy()


def _set_display_flags(node) -> None:
    for method_name in ("setDisplayFlag", "setRenderFlag"):
        method = getattr(node, method_name, None)
        if method is not None:
            method(True)


def _safe_node_name(prefix: str, text: str) -> str:
    suffix = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    return f"{prefix}_{suffix}" if suffix else prefix
