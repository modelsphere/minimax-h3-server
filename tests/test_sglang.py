from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from h3_service.config import Settings
from h3_service.sglang import BackendError, SGLangClient
from h3_service.storage import SharedStorage


def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url="postgresql://unused",
        storage_root=tmp_path,
        sglang_url="http://sglang:30010",
        start_workers=False,
    )


def test_default_task_lease_is_five_minutes(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_STORAGE_ROOT", str(tmp_path))
    monkeypatch.delenv("H3_TASK_LEASE_SECONDS", raising=False)
    monkeypatch.delenv("H3_MAX_ATTEMPTS", raising=False)

    config = Settings.from_env()
    assert config.lease_seconds == 300
    assert config.max_attempts == 5


def test_default_backend_ready_timeout_is_two_seconds(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_STORAGE_ROOT", str(tmp_path))
    monkeypatch.delenv("H3_BACKEND_READY_TIMEOUT_SECONDS", raising=False)

    assert Settings.from_env().backend_ready_timeout_seconds == 2


def test_ref2va_sampling_defaults_to_backend_base_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("H3_SERVICE_CAPABILITY", "ref2va")
    for name in ("H3_REF2VA_NUM_INFERENCE_STEPS", "H3_REF2VA_FLOW_SHIFT", "H3_REF2VA_AUDIO_FLOW_SHIFT"):
        monkeypatch.delenv(name, raising=False)

    config = Settings.from_env()
    config.validate()

    assert config.ref2va_sampling == {}


def test_ref2va_sampling_environment_overrides_reach_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("H3_SERVICE_CAPABILITY", "ref2va")
    monkeypatch.setenv("H3_REF2VA_NUM_INFERENCE_STEPS", "50")
    monkeypatch.setenv("H3_REF2VA_FLOW_SHIFT", "12.0")
    monkeypatch.setenv("H3_REF2VA_AUDIO_FLOW_SHIFT", "3.0")
    config = Settings.from_env()
    config.validate()
    task = {
        "task_mode": "ref2va", "prompt": "move", "duration": 5, "ratio": "16:9", "seed": 3,
        "assets": [{"asset_type": "input", "role": "reference_image", "relative_path": "task/ref.png"}],
    }

    payload = SGLangClient(config, SharedStorage(tmp_path, "/data/minimax-h3", False, 2)).build_payload(task)

    assert payload["num_inference_steps"] == 50
    assert payload["flow_shift"] == 12.0
    assert payload["audio_flow_shift"] == 3.0
    assert payload["quality"] == "lossless"


@pytest.mark.parametrize("steps", [0, 1, -1, True, 2.5])
def test_ref2va_rejects_invalid_step_counts(tmp_path, steps):
    config = replace(settings(tmp_path), capability="ref2va", ref2va_num_inference_steps=steps,
                     ref2va_flow_shift=6.0, ref2va_audio_flow_shift=3.0)
    with pytest.raises(RuntimeError, match="H3_REF2VA_NUM_INFERENCE_STEPS"):
        config.validate()


@pytest.mark.parametrize("field, env_name", [
    ("ref2va_flow_shift", "H3_REF2VA_FLOW_SHIFT"),
    ("ref2va_audio_flow_shift", "H3_REF2VA_AUDIO_FLOW_SHIFT"),
])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_ref2va_rejects_invalid_shifts(tmp_path, field, env_name, value):
    values = dict(ref2va_num_inference_steps=9, ref2va_flow_shift=6.0, ref2va_audio_flow_shift=3.0)
    values[field] = value
    config = replace(settings(tmp_path), capability="ref2va", **values)
    with pytest.raises(RuntimeError, match=env_name):
        config.validate()


@pytest.mark.parametrize("mode", ["t2va", "fl2va"])
def test_ref2va_sampling_overrides_do_not_change_fl2va_payload(tmp_path, mode):
    config = replace(settings(tmp_path), ref2va_num_inference_steps=5,
                     ref2va_flow_shift=8.0, ref2va_audio_flow_shift=4.0)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    task = {"task_mode": mode, "prompt": "move", "duration": 5, "ratio": "16:9", "seed": 3, "assets": []}
    if mode == "fl2va":
        task["assets"] = [{"asset_type": "input", "role": "first_frame", "relative_path": "task/first.png"}]

    payload = SGLangClient(config, storage).build_payload(task)

    assert payload == SGLangClient(settings(tmp_path), storage).build_payload(task)
    assert payload["num_inference_steps"] == 50
    assert payload["flow_shift"] == 12.0
    assert payload["audio_flow_shift"] == 3.0


@pytest.mark.parametrize("status_code, expected", [(200, True), (503, False), (500, False)])
def test_backend_readiness_requires_health_200(monkeypatch, tmp_path, status_code, expected):
    observed = {}

    def fake_get(url, timeout):
        observed.update(url=url, timeout=timeout)
        return httpx.Response(status_code)

    monkeypatch.setattr(httpx, "get", fake_get)
    client = SGLangClient(
        settings(tmp_path),
        SharedStorage(tmp_path, "/data/minimax-h3", False, 2),
        timeout=1.25,
    )

    assert client.is_ready() is expected
    assert observed == {"url": "http://sglang:30010/health", "timeout": 1.25}


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_backend_readiness_treats_transport_errors_as_not_ready(monkeypatch, tmp_path, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(httpx, "get", fail)
    client = SGLangClient(settings(tmp_path), SharedStorage(tmp_path, "/data/minimax-h3", False, 2))

    assert client.is_ready() is False


def test_first_last_frames_map_to_sglang_frame_indexes(tmp_path):
    config = settings(tmp_path)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    task = {
        "task_mode": "fl2va",
        "prompt": "move",
        "duration": 5,
        "ratio": "adaptive",
        "seed": 42,
        "assets": [
            {"asset_type": "input", "role": "last_frame", "relative_path": "task/last.png"},
            {"asset_type": "input", "role": "first_frame", "relative_path": "task/first.png"},
            {"asset_type": "thumbnail", "role": "first_frame", "relative_path": "task/thumb.jpg"},
        ],
    }
    payload = SGLangClient(config, storage).build_payload(task)
    assert payload["model"] == "/model"
    assert payload["task"] == "fl2va"
    assert [item["frame_index"] for item in payload["conditions"]] == [0, -1]
    assert payload["conditions"][0]["uri"] == "file:///data/minimax-h3/task/first.png"
    assert payload["target"]["aspect_ratio"] == "auto"


def test_t2v_keeps_requested_ratio(tmp_path):
    config = settings(tmp_path)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    task = {"task_mode": "t2va", "prompt": "move", "duration": 5, "ratio": "9:16", "seed": 3, "assets": []}
    payload = SGLangClient(config, storage).build_payload(task)
    assert payload["conditions"] == []
    assert payload["target"]["aspect_ratio"] == "9:16"


@pytest.mark.parametrize("turbo", [False, True])
def test_ref2va_payload_matches_sglang_interface_and_source_order(tmp_path, turbo):
    config = replace(settings(tmp_path), capability="ref2va")
    if turbo:
        config = replace(config, ref2va_num_inference_steps="9.0",
                         ref2va_flow_shift="6.0", ref2va_audio_flow_shift="3.0")
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    task = {
        "task_mode": "ref2va",
        "prompt": "An otter takes the orange.",
        "duration": 5,
        "ratio": "adaptive",
        "seed": 0,
        "assets": [
            {"asset_id": 4, "asset_type": "input", "role": "reference_audio", "source_index": 3, "relative_path": "task/input.wav"},
            {"asset_id": 2, "asset_type": "input", "role": "reference_image", "source_index": 1, "relative_path": "task/input.jpg"},
            {"asset_id": 3, "asset_type": "input", "role": "reference_video", "source_index": 2, "relative_path": "task/input.mp4"},
            {"asset_id": 5, "asset_type": "thumbnail", "role": "reference_video", "source_index": 2, "relative_path": "task/thumb.jpg"},
        ],
    }

    payload = SGLangClient(config, storage).build_payload(task)

    expected = {
        "model": "/model",
        "prompt": "An otter takes the orange.",
        "task": "ref2va",
        "conditions": [
            {"type": "image", "uri": "file:///data/minimax-h3/task/input.jpg", "role": "reference"},
            {"type": "video", "uri": "file:///data/minimax-h3/task/input.mp4", "role": "reference"},
            {"type": "audio", "uri": "file:///data/minimax-h3/task/input.wav", "role": "reference"},
        ],
        "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.0},
        "seed": 0,
    }
    if turbo:
        expected.update(num_inference_steps=9, flow_shift=6.0, audio_flow_shift=3.0, quality="lossless")
    assert payload == expected


@pytest.mark.parametrize("frame_roles, frame_indices", [
    (["first_frame"], [0]),
    (["last_frame"], [-1]),
    (["last_frame", "first_frame"], [0, -1]),
])
@pytest.mark.parametrize("turbo", [False, True])
def test_ref2va_keyframes_are_ordered_without_reordering_references(tmp_path, frame_roles, frame_indices, turbo):
    config = replace(settings(tmp_path), capability="ref2va")
    if turbo:
        config = replace(config, ref2va_num_inference_steps=9,
                         ref2va_flow_shift=6.0, ref2va_audio_flow_shift=3.0)
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    task = {
        "task_mode": "ref2va", "prompt": "Follow <Picture 1> and <Video 1>.",
        "duration": 5, "ratio": "9:16", "seed": 0,
        "assets": [
            {"asset_id": 5, "asset_type": "input", "role": "reference_audio", "source_index": 5, "relative_path": "task/ref.wav"},
            *[
                {"asset_type": "input", "role": role, "relative_path": f"task/{role}.png"}
                for role in frame_roles
            ],
            {"asset_id": 4, "asset_type": "input", "role": "reference_video", "source_index": 4, "relative_path": "task/ref.mp4"},
            {"asset_id": 1, "asset_type": "input", "role": "reference_image", "source_index": 1, "relative_path": "task/ref.png"},
            {"asset_type": "thumbnail", "role": "first_frame", "relative_path": "task/thumb.jpg"},
        ],
    }

    payload = SGLangClient(config, storage).build_payload(task)

    keyframes = payload["conditions"][:len(frame_roles)]
    assert keyframes == [
        {"type": "image", "role": "keyframe", "frame_index": frame_index,
         "uri": f"file:///data/minimax-h3/task/{'first_frame' if frame_index == 0 else 'last_frame'}.png"}
        for frame_index in frame_indices
    ]
    assert payload["conditions"][len(frame_roles):] == [
        {"type": "image", "role": "reference", "uri": "file:///data/minimax-h3/task/ref.png"},
        {"type": "video", "role": "reference", "uri": "file:///data/minimax-h3/task/ref.mp4"},
        {"type": "audio", "role": "reference", "uri": "file:///data/minimax-h3/task/ref.wav"},
    ]
    assert payload["task"] == "ref2va"
    assert payload["target"] == {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 5.0}
    if turbo:
        assert payload["num_inference_steps"] == 9
        assert payload["flow_shift"] == 6.0
        assert payload["audio_flow_shift"] == 3.0
        assert payload["quality"] == "lossless"
    else:
        assert not {"num_inference_steps", "flow_shift", "audio_flow_shift", "quality"} & payload.keys()


def test_legacy_sglang_model_env_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("H3_SGLANG_MODEL", "/legacy-model")
    config = Settings.from_env()
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    task = {"task_mode": "t2va", "prompt": "move", "duration": 5, "ratio": "16:9", "seed": 3, "assets": []}

    payload = SGLangClient(config, storage).build_payload(task)

    assert not hasattr(config, "sglang_model")
    assert payload["model"] == "/model"


def test_status_connection_reset_is_retryable(monkeypatch, tmp_path):
    client = SGLangClient(settings(tmp_path), SharedStorage(tmp_path, "/data/minimax-h3", False, 2))
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ReadError("reset")))

    with pytest.raises(BackendError) as error:
        client.status("job-1")

    assert error.value.retryable is True
    assert error.value.ambiguous is False


def test_submission_connection_reset_is_ambiguous(monkeypatch, tmp_path):
    client = SGLangClient(settings(tmp_path), SharedStorage(tmp_path, "/data/minimax-h3", False, 2))
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ReadError("reset")))
    task = {"task_mode": "t2va", "prompt": "move", "duration": 5, "ratio": "16:9", "seed": 3, "assets": []}

    with pytest.raises(BackendError) as error:
        client.submit(task)

    assert error.value.ambiguous is True
    assert error.value.retryable is False
