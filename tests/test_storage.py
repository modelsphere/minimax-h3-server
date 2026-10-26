import base64
import io
import wave

import pytest
from PIL import Image

from h3_service.models import APIError, MediaInput
from h3_service.storage import SharedStorage, StoredAsset, validate_remote_url


def image_data_url() -> str:
    output = io.BytesIO()
    Image.new("RGB", (320, 256), "teal").save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def wav_data() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 16_000 * 2)
    return output.getvalue()


def test_persist_input_and_thumbnail_on_mounted_layout(tmp_path):
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    storage.init()
    asset, thumbnail = storage.persist_input("00000000-0000-0000-0000-000000000001", MediaInput("image", "first_frame", image_data_url()))
    assert storage.resolve(asset.relative_path).read_bytes().startswith(b"\x89PNG")
    assert thumbnail is not None
    assert storage.resolve(thumbnail.relative_path).is_file()
    assert storage.sglang_uri(asset.relative_path) == f"file:///data/minimax-h3/{asset.relative_path}"


def test_multiple_reference_images_use_distinct_paths_and_indexes(tmp_path):
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    storage.init()
    first, first_thumbnail = storage.persist_input(
        "task-1", MediaInput("image", "reference_image", image_data_url(), 1)
    )
    second, second_thumbnail = storage.persist_input(
        "task-1", MediaInput("image", "reference_image", image_data_url(), 2)
    )

    assert first.source_index == 1
    assert second.source_index == 2
    assert first.relative_path != second.relative_path
    assert first_thumbnail is not None and second_thumbnail is not None
    assert first_thumbnail.relative_path != second_thumbnail.relative_path


def test_wav_validation_records_duration():
    media_type, suffix, metadata = SharedStorage._validate_audio(wav_data())

    assert media_type == "audio/wav"
    assert suffix == ".wav"
    assert metadata["audio_codec"] == "pcm_s16le"
    assert metadata["duration_seconds"] == pytest.approx(2.0, abs=0.01)


def test_invalid_video_is_rejected():
    with pytest.raises(APIError, match="valid supported video"):
        SharedStorage._validate_video(b"not a video")


def test_reference_total_durations_are_limited():
    assets = [
        StoredAsset("input", "reference_video", "a.mp4", "video/mp4", 1, metadata={"duration_seconds": 8}),
        StoredAsset("input", "reference_video", "b.mp4", "video/mp4", 1, metadata={"duration_seconds": 8}),
    ]

    with pytest.raises(APIError, match="total reference video duration"):
        SharedStorage.validate_reference_totals(assets)


def test_asset_path_cannot_escape_mount(tmp_path):
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    with pytest.raises(ValueError):
        storage.resolve("../../etc/passwd")


@pytest.mark.parametrize("url", ["http://127.0.0.1/a", "http://[::1]/a", "file:///etc/passwd"])
def test_private_and_non_http_urls_are_rejected(url):
    with pytest.raises(Exception):
        validate_remote_url(url)


def test_delete_prunes_empty_directories(tmp_path):
    """删完素材要把空目录一起收掉 —— 否则每个任务留 3~4 个空目录,
    长期在共享盘上积累 inode。"""
    storage = SharedStorage(tmp_path, "/data/minimax-h3", False, 2)
    storage.init()

    rel = "2026/09/01/task-abc/output/video.mp4"
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"data")
    sibling = tmp_path / "2026/09/01/task-abc/inputs/first.png"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"png")

    storage.delete(rel)

    assert not target.exists()
    assert not target.parent.exists(), "output/ 空了应该被删掉"
    # 还有别的素材没删,任务目录和日期目录要留着
    assert sibling.exists()
    assert sibling.parent.exists()
    assert (tmp_path / "2026/09/01/task-abc").exists()

    storage.delete("2026/09/01/task-abc/inputs/first.png")
    assert not (tmp_path / "2026/09/01").exists(), "整条链空了应一路收到 root 下"
    assert tmp_path.exists(), "storage root 本身不能删"
