from dataclasses import FrozenInstanceError, replace

import pytest

from h3_service.config import Settings


FIELDS = ("num_inference_steps", "flow_shift", "audio_flow_shift")


@pytest.fixture
def ref_settings(tmp_path):
    return Settings(database_url="postgresql://unused", storage_root=tmp_path,
                    sglang_url="http://sglang:30010", capability="ref2va")


@pytest.mark.parametrize("values", [(None, None, None), ("", "  ", "\t")])
def test_unset_and_blank_sampling_restore_base(ref_settings, values):
    config = replace(ref_settings, **dict(zip((f"ref2va_{key}" for key in FIELDS), values)))
    config.validate()
    assert config.ref2va_sampling == {}


@pytest.mark.parametrize("steps, expected", [(9, 9), (9.0, 9), (" 9.0 ", 9), ("9e0", 9), ("1e6", 1000000)])
def test_steps_accept_exact_integer_numeric_forms(ref_settings, steps, expected):
    config = replace(ref_settings, ref2va_num_inference_steps=steps,
                     ref2va_flow_shift="6e0", ref2va_audio_flow_shift=" 3.0 ")
    config.validate()
    assert config.ref2va_sampling == dict(num_inference_steps=expected, flow_shift=6.0, audio_flow_shift=3.0)
    assert type(config.ref2va_sampling["num_inference_steps"]) is int
    with pytest.raises(FrozenInstanceError):
        config.ref2va_num_inference_steps = 50


@pytest.mark.parametrize("values", [("9", None, None), (None, "6", "3"), ("9", "", "3"), (None, None, "3")])
def test_partial_sampling_is_rejected(ref_settings, values):
    config = replace(ref_settings, **dict(zip((f"ref2va_{key}" for key in FIELDS), values)))
    with pytest.raises(RuntimeError, match="must be set together") as error:
        config.validate()
    for field in FIELDS:
        assert f"H3_REF2VA_{field.upper()}" in str(error.value)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [True, False, "bad", "nan", "inf", "-inf", "0", "-1", [], {}])
def test_invalid_values_report_the_variable(ref_settings, field, value):
    fields = dict(ref2va_num_inference_steps="9", ref2va_flow_shift="6", ref2va_audio_flow_shift="3")
    fields[f"ref2va_{field}"] = value
    with pytest.raises(RuntimeError, match=f"H3_REF2VA_{field.upper()}"):
        replace(ref_settings, **fields).validate()


@pytest.mark.parametrize("value", ["1", "2.5", "NaN", "Infinity", "1e-1"])
def test_invalid_step_strings(ref_settings, value):
    with pytest.raises(RuntimeError, match="H3_REF2VA_NUM_INFERENCE_STEPS"):
        replace(ref_settings, ref2va_num_inference_steps=value,
                ref2va_flow_shift=6, ref2va_audio_flow_shift=3).validate()


@pytest.mark.parametrize("capability", ["fl2va", "ref2va"])
def test_from_env_parses_only_for_ref2va(monkeypatch, capability):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_SERVICE_CAPABILITY", capability)
    for field in FIELDS:
        monkeypatch.setenv(f"H3_REF2VA_{field.upper()}", "invalid")
    config = Settings.from_env()  # Never raises a raw int()/float() ValueError.
    if capability == "ref2va":
        with pytest.raises(RuntimeError, match="H3_REF2VA_NUM_INFERENCE_STEPS"):
            config.validate()
    else:
        config.validate()
        assert config.ref2va_sampling == {}


def test_from_env_normalizes_decimal_steps(monkeypatch):
    monkeypatch.setenv("H3_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("H3_SERVICE_CAPABILITY", "ref2va")
    for field, value in zip(FIELDS, ("9.0", "6e0", "3.0")):
        monkeypatch.setenv(f"H3_REF2VA_{field.upper()}", value)
    config = Settings.from_env()
    config.validate()
    assert config.ref2va_sampling == dict(num_inference_steps=9, flow_shift=6.0, audio_flow_shift=3.0)
