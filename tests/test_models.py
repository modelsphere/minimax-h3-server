import pytest

from h3_service.models import APIError, parse_video_request


def request(content, **overrides):
    value = {
        "model": "MiniMax-H3",
        "content": content,
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
    }
    value.update(overrides)
    return value


def test_t2v_maps_to_fl2va_capability():
    parsed = parse_video_request(request([{"type": "text", "text": " hello "}]))
    assert parsed.mode == "t2va"
    assert parsed.required_capability == "fl2va"
    assert parsed.prompt == "hello"


@pytest.mark.parametrize("model", ["/model", "MiniMax-H3-Max", "other"])
def test_only_official_minimax_h3_model_is_accepted(model):
    with pytest.raises(APIError) as error:
        parse_video_request(request([{"type": "text", "text": "hello"}], model=model))
    assert error.value.status_code == 400


def test_first_and_last_frame_force_adaptive_ratio():
    parsed = parse_video_request(request([
        {"type": "text", "text": "move"},
        {"type": "image_url", "role": "first_frame", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "image_url", "role": "last_frame", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]))
    assert parsed.mode == "fl2va"
    assert parsed.ratio == "adaptive"
    assert [item.role for item in parsed.media] == ["first_frame", "last_frame"]


def test_ref_interface_is_recognized_and_separate():
    parsed = parse_video_request(request([
        {"type": "text", "text": "follow this character"},
        {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://example.com/ref.jpg"}},
    ]))
    assert parsed.mode == "ref2va"
    assert parsed.required_capability == "ref2va"


def test_ref_image_video_and_audio_keep_request_order():
    parsed = parse_video_request(request([
        {"type": "text", "text": "follow these references"},
        {"type": "video_url", "role": "reference_video", "video_url": {"url": "https://example.com/ref.mp4"}},
        {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://example.com/ref.jpg"}},
        {"type": "audio_url", "role": "reference_audio", "audio_url": {"url": "https://example.com/ref.wav"}},
    ], ratio="adaptive"))

    assert [(item.media_type, item.role, item.source_index) for item in parsed.media] == [
        ("video", "reference_video", 1),
        ("image", "reference_image", 2),
        ("audio", "reference_audio", 3),
    ]


def test_reference_audio_requires_image_or_video():
    with pytest.raises(APIError) as error:
        parse_video_request(request([
            {"type": "text", "text": "listen"},
            {"type": "audio_url", "role": "reference_audio", "audio_url": {"url": "https://example.com/ref.wav"}},
        ], ratio="adaptive"))
    assert "requires a reference image or video" in error.value.message


def test_ref_total_material_limit_is_enforced():
    content = [{"type": "text", "text": "follow these references"}]
    content.extend(
        {"type": "image_url", "role": "reference_image", "image_url": {"url": f"https://example.com/{index}.jpg"}}
        for index in range(9)
    )
    content.extend(
        {"type": "video_url", "role": "reference_video", "video_url": {"url": f"https://example.com/{index}.mp4"}}
        for index in range(3)
    )
    content.append({"type": "audio_url", "role": "reference_audio", "audio_url": {"url": "https://example.com/ref.wav"}})

    with pytest.raises(APIError) as error:
        parse_video_request(request(content, ratio="adaptive"))
    assert "at most 12 reference media files" in error.value.message


@pytest.mark.parametrize("changes", [
    {"resolution": "2K"},
    {"duration": 16},
    {"ratio": "adaptive"},
])
def test_unsupported_t2v_options_are_rejected(changes):
    with pytest.raises(APIError) as error:
        parse_video_request(request([{"type": "text", "text": "hello"}], **changes))
    assert error.value.status_code == 400


@pytest.mark.parametrize("frame_roles", [
    ["first_frame"], ["last_frame"], ["last_frame", "first_frame"],
])
@pytest.mark.parametrize("reference_type, reference_role", [
    ("image_url", "reference_image"), ("video_url", "reference_video"),
])
@pytest.mark.parametrize("ratio", ["adaptive", "9:16"])
def test_frames_and_references_use_ref2va(frame_roles, reference_type, reference_role, ratio):
    content = [{"type": "text", "text": "hello"}]
    content.extend(
        {"type": "image_url", "role": role, "image_url": {"url": f"https://example.com/{role}.jpg"}}
        for role in frame_roles
    )
    content.append({
        "type": reference_type, "role": reference_role,
        reference_type: {"url": "https://example.com/ref"},
    })

    parsed = parse_video_request(request(content, ratio=ratio))

    assert parsed.mode == "ref2va"
    assert parsed.required_capability == "ref2va"
    assert parsed.ratio == ratio
    assert [item.role for item in parsed.media] == frame_roles + [reference_role]


@pytest.mark.parametrize("role", ["first_frame", "last_frame"])
def test_ref2va_rejects_duplicate_frames(role):
    with pytest.raises(APIError) as error:
        parse_video_request(request([
            {"type": "text", "text": "hello"},
            {"type": "image_url", "role": "reference_image", "image_url": {"url": "https://example.com/ref.jpg"}},
            *[
                {"type": "image_url", "role": role, "image_url": {"url": f"https://example.com/{index}.jpg"}}
                for index in range(2)
            ],
        ]))
    assert "at most one first frame and one last frame" in error.value.message


def test_keyframe_does_not_replace_visual_reference_for_audio():
    with pytest.raises(APIError) as error:
        parse_video_request(request([
            {"type": "text", "text": "hello"},
            {"type": "image_url", "role": "first_frame", "image_url": {"url": "https://example.com/first.jpg"}},
            {"type": "audio_url", "role": "reference_audio", "audio_url": {"url": "https://example.com/ref.wav"}},
        ]))
    assert "requires a reference image or video" in error.value.message
