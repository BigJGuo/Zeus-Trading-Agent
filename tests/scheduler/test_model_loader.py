"""Tests for the model-loader helpers in `zeus.scheduler.runner`.

The regression target is the `'params'` KeyError that `_load_latest_model`
used to raise when a CrossHorizonGatedPredictor bundle was promoted with
no `__model_class__` config hint and no `model_class.txt` sidecar: the
legacy default `XGBReturnPredictor.load(...)` would crash on `payload["params"]`.
`_resolve_model_class` now sniffs the joblib payload to recover.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest


def _write_payload(tmp: Path, name: str, payload: dict) -> Path:
    path = tmp / name
    joblib.dump(payload, path)
    return path


# ─── Payload sniffing ─────────────────────────────────────────────────────────


def test_detect_cross_horizon_from_primary_cls_key(tmp_path):
    from zeus.scheduler.runner import _detect_class_from_payload

    p = _write_payload(tmp_path, "model.joblib", {
        "primary_cls": "zeus.models.return_predictor:XGBReturnPredictor",
        "primary_path": "ignored",
        "mag_floor": 0.0,
        "rank_floor": 0.95,
    })
    assert _detect_class_from_payload(p) == (
        "zeus.models.rank_gated_predictor:CrossHorizonGatedPredictor"
    )


def test_detect_xgb_from_params_key(tmp_path):
    from zeus.scheduler.runner import _detect_class_from_payload

    p = _write_payload(tmp_path, "model.joblib", {
        "params": {"max_depth": 6},
        "model": object(),
        "feature_names": ["a", "b"],
    })
    assert _detect_class_from_payload(p) == (
        "zeus.models.return_predictor:XGBReturnPredictor"
    )


def test_detect_rank_gated_from_rank_floor_key(tmp_path):
    """A RankMagnitudeGated bundle has `rank_floor` but NOT `primary_cls` —
    must NOT misroute to CrossHorizon. Order in `_PAYLOAD_KEY_TO_MODEL_PATH`
    matters: primary_cls is checked first."""
    from zeus.scheduler.runner import _detect_class_from_payload

    p = _write_payload(tmp_path, "model.joblib", {
        "rank_floor": 0.95,
        "mag_floor": 0.0,
        "base_model": object(),
    })
    assert _detect_class_from_payload(p) == (
        "zeus.models.rank_gated_predictor:RankMagnitudeGatedPredictor"
    )


def test_detect_ensemble_from_members_key(tmp_path):
    from zeus.scheduler.runner import _detect_class_from_payload

    p = _write_payload(tmp_path, "model.joblib", {
        "members": [],
        "weights": [],
    })
    assert _detect_class_from_payload(p) == (
        "zeus.models.ensemble_return_predictor:EnsembleReturnPredictor"
    )


def test_detect_returns_none_when_no_marker(tmp_path):
    from zeus.scheduler.runner import _detect_class_from_payload

    p = _write_payload(tmp_path, "model.joblib", {"unknown_key": 1})
    assert _detect_class_from_payload(p) is None


def test_detect_handles_missing_file(tmp_path):
    """A path that doesn't exist must not raise — return None and let the
    caller fall back to its default class."""
    from zeus.scheduler.runner import _detect_class_from_payload

    assert _detect_class_from_payload(tmp_path / "nonexistent.joblib") is None


def test_detect_handles_non_dict_payload(tmp_path):
    from zeus.scheduler.runner import _detect_class_from_payload

    p = _write_payload(tmp_path, "model.joblib", ["not", "a", "dict"])
    assert _detect_class_from_payload(p) is None


# ─── Class resolution priority ────────────────────────────────────────────────


def test_config_hint_wins_over_sidecar_and_payload(tmp_path):
    """Priority: config.__model_class__ > model_class.txt > payload sniff."""
    from zeus.scheduler.runner import _resolve_model_class

    # Sidecar says one class, payload says another, config beats both.
    (tmp_path / "model_class.txt").write_text("zeus.models.return_predictor:XGBReturnPredictor")
    p = _write_payload(tmp_path, "model.joblib", {"primary_cls": "x:Y"})
    row = SimpleNamespace(config={"__model_class__": "zeus.models.meta_labeler:MetaGatedPredictor"})

    assert _resolve_model_class(row, tmp_path, p) == (
        "zeus.models.meta_labeler:MetaGatedPredictor"
    )


def test_sidecar_wins_over_payload_when_no_config_hint(tmp_path):
    from zeus.scheduler.runner import _resolve_model_class

    (tmp_path / "model_class.txt").write_text("zeus.models.return_predictor:XGBReturnPredictor")
    p = _write_payload(tmp_path, "model.joblib", {"primary_cls": "x:Y"})
    row = SimpleNamespace(config=None)

    assert _resolve_model_class(row, tmp_path, p) == (
        "zeus.models.return_predictor:XGBReturnPredictor"
    )


def test_payload_sniff_used_when_neither_config_nor_sidecar(tmp_path):
    """The exact regression scenario — older bundles have no hint, must
    be detectable from the joblib's structure."""
    from zeus.scheduler.runner import _resolve_model_class

    p = _write_payload(tmp_path, "model.joblib", {
        "primary_cls": "zeus.models.return_predictor:XGBReturnPredictor",
        "mag_floor": 0.0,
        "rank_floor": 0.95,
    })
    row = SimpleNamespace(config=None)

    assert _resolve_model_class(row, tmp_path, p) == (
        "zeus.models.rank_gated_predictor:CrossHorizonGatedPredictor"
    )


def test_resolve_returns_none_when_all_methods_fail(tmp_path):
    from zeus.scheduler.runner import _resolve_model_class

    p = _write_payload(tmp_path, "model.joblib", {"mystery": 1})
    row = SimpleNamespace(config={})

    assert _resolve_model_class(row, tmp_path, p) is None
