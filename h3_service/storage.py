from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import os
import socket
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urljoin, urlparse

import av
import httpx
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .models import APIError, MediaInput


register_heif_opener()

IMAGE_MAX_BYTES = 30 * 1024 * 1024
VIDEO_MAX_BYTES = 50 * 1024 * 1024
AUDIO_MAX_BYTES = 15 * 1024 * 1024
IMAGE_MIN_DIMENSION = 256
IMAGE_MAX_DIMENSION = 5760
IMAGE_RATIO_MIN = 0.4
IMAGE_RATIO_MAX = 2.5
MEDIA_MIN_DURATION = 2.0
MEDIA_MAX_DURATION = 15.0
VIDEO_MIN_FPS = 23.976
VIDEO_MAX_FPS = 60.0
ALLOWED_IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
    "HEIF": ("image/heif", ".heif"),
    "HEIC": ("image/heic", ".heic"),
}
ALLOWED_VIDEO_CODECS = {"h264", "hevc"}
ALLOWED_VIDEO_AUDIO_CODECS = {"aac", "mp3"}


@dataclass(frozen=True)
class StoredAsset:
    asset_type: str
    role: str
    relative_path: str
    media_type: str
    byte_size: int
    width: int | None = None
    height: int | None = None
    source_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _media_error(message: str) -> APIError:
    return APIError(400, "bad_request_error", message)


def _is_forbidden_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any((
        ip.is_private,
        ip.is_loopback,
        ip.is_link_local,
        ip.is_multicast,
        ip.is_reserved,
        ip.is_unspecified,
    ))


def validate_remote_url(url: str, allow_private: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise _media_error("media and callback URLs must use http or https")
    if parsed.username or parsed.password:
        raise _media_error("URLs containing credentials are not allowed")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise _media_error("URL contains an invalid port") from exc
    if allow_private:
        return
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, port)}
    except (OSError, ValueError) as exc:
        raise _media_error(f"cannot resolve URL host: {parsed.hostname}") from exc
    if not addresses or any(_is_forbidden_ip(address) for address in addresses):
        raise _media_error("private, loopback and link-local URLs are not allowed")


def _decode_data_url(url: str) -> tuple[bytes, str]:
    try:
        header, encoded = url.split(",", 1)
    except ValueError as exc:
        raise _media_error("invalid data URL") from exc
    if not header.startswith("data:"):
        raise _media_error("invalid data URL")
    meta = header[5:].split(";")
    media_type = meta[0] or "text/plain"
    try:
        data = base64.b64decode(encoded, validate=True) if "base64" in meta[1:] else unquote_to_bytes(encoded)
    except (binascii.Error, ValueError) as exc:
        raise _media_error("invalid data URL encoding") from exc
    return data, media_type


def _stream_duration(container: av.container.InputContainer, stream: av.stream.Stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    raise _media_error("input media duration could not be determined")


def _check_dimensions(width: int, height: int, kind: str) -> None:
    if not (
        IMAGE_MIN_DIMENSION <= width <= IMAGE_MAX_DIMENSION
        and IMAGE_MIN_DIMENSION <= height <= IMAGE_MAX_DIMENSION
    ):
        raise _media_error(f"{kind} dimensions must be between 256 and 5760 pixels")
    ratio = width / height
    if not IMAGE_RATIO_MIN <= ratio <= IMAGE_RATIO_MAX:
        raise _media_error(f"{kind} aspect ratio must be between 0.4 and 2.5")


class SharedStorage:
    def __init__(self, root: Path, sglang_media_root: str, allow_private_urls: bool, timeout: float):
        self.root = root.resolve()
        self.sglang_media_root = sglang_media_root.rstrip("/")
        self.allow_private_urls = allow_private_urls
        self.timeout = timeout

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.root, os.W_OK):
            raise RuntimeError(f"storage root is not writable: {self.root}")

    def task_prefix(self, task_id: str, created_at: datetime | None = None) -> Path:
        value = created_at or datetime.now(timezone.utc)
        return Path(value.strftime("%Y/%m/%d")) / task_id

    def persist_input(
        self,
        task_id: str,
        media: MediaInput,
        created_at: datetime | None = None,
    ) -> tuple[StoredAsset, StoredAsset | None]:
        limits = {"image": IMAGE_MAX_BYTES, "video": VIDEO_MAX_BYTES, "audio": AUDIO_MAX_BYTES}
        accepts = {"image": "image/*", "video": "video/*", "audio": "audio/*"}
        data, _claimed_type = self._read_url(media.url, limits[media.media_type], accepts[media.media_type])

        thumbnail_image: Image.Image | None = None
        if media.media_type == "image":
            image, media_type, suffix = self._validate_image(data)
            width, height = image.size
            metadata = {"kind": "image"}
            thumbnail_image = image
        elif media.media_type == "video":
            media_type, suffix, width, height, metadata, thumbnail_image = self._validate_video(data)
        else:
            media_type, suffix, metadata = self._validate_audio(data)
            width = height = None

        prefix = self.task_prefix(task_id, created_at)
        role_name = media.role.replace("_", "-")
        if media.role.startswith("reference_"):
            role_name = f"{role_name}-{media.source_index:02d}"
        relative = prefix / "inputs" / f"{role_name}{suffix}"
        self._atomic_write(relative, data)
        asset = StoredAsset(
            "input",
            media.role,
            relative.as_posix(),
            media_type,
            len(data),
            width,
            height,
            media.source_index,
            metadata,
        )

        if thumbnail_image is None:
            return asset, None
        try:
            thumbnail_relative = prefix / "thumbnail" / f"{role_name}.jpg"
            thumbnail_data, thumb_width, thumb_height = self._thumbnail(thumbnail_image)
            self._atomic_write(thumbnail_relative, thumbnail_data)
        except Exception:
            self.delete(relative.as_posix())
            raise
        thumbnail = StoredAsset(
            "thumbnail",
            media.role,
            thumbnail_relative.as_posix(),
            "image/jpeg",
            len(thumbnail_data),
            thumb_width,
            thumb_height,
            media.source_index,
            {"kind": "thumbnail"},
        )
        return asset, thumbnail

    @staticmethod
    def validate_reference_totals(assets: list[StoredAsset]) -> None:
        inputs = [asset for asset in assets if asset.asset_type == "input"]
        video_seconds = sum(
            float(asset.metadata.get("duration_seconds", 0))
            for asset in inputs
            if asset.role == "reference_video"
        )
        audio_seconds = sum(
            float(asset.metadata.get("duration_seconds", 0))
            for asset in inputs
            if asset.role == "reference_audio"
        )
        if video_seconds > MEDIA_MAX_DURATION + 1e-3:
            raise _media_error("total reference video duration must not exceed 15 seconds")
        if audio_seconds > MEDIA_MAX_DURATION + 1e-3:
            raise _media_error("total reference audio duration must not exceed 15 seconds")

    def output_relative_path(self, task: dict) -> Path:
        created = task.get("created_at")
        return self.task_prefix(task["task_id"], created) / "output" / "video.mp4"

    def sglang_uri(self, relative_path: str) -> str:
        return f"file://{self.sglang_media_root}/{relative_path.lstrip('/')}"

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("asset path escapes storage root") from exc
        return candidate

    def delete(self, relative_path: str | None) -> None:
        if not relative_path:
            return
        path = self.resolve(relative_path)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._prune_empty_dirs(path.parent)

    def _prune_empty_dirs(self, directory: Path) -> None:
        """自底向上删掉变空的目录,到 storage root 为止(root 本身不删)。

        不这么做的话,素材清理完会留下每任务 3~4 个空目录
        (<task_id>/inputs、/thumbnail、/output 以及 <task_id> 自己),
        长期在共享盘(CephFS)上积累 inode。
        并发下别的实例可能正往同一目录写文件,所以用 rmdir 而不是 rmtree:
        目录非空时 OSError,忽略即可 —— 下一轮清理会再来。
        """
        root = self.root.resolve()
        current = directory.resolve()
        while current != root and root in current.parents:
            try:
                current.rmdir()
            except OSError:
                return          # 非空 / 已不存在 / 无权限 —— 停在这里
            current = current.parent

    def write_response_stream(self, relative_path: Path, chunks) -> StoredAsset:
        target = self.resolve(relative_path.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
        size = 0
        try:
            with os.fdopen(fd, "wb") as handle:
                for chunk in chunks:
                    if chunk:
                        handle.write(chunk)
                        size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return StoredAsset("output", "generated_video", relative_path.as_posix(), "video/mp4", size)

    def _read_url(self, url: str, max_bytes: int, accept: str) -> tuple[bytes, str]:
        if url.startswith("data:"):
            data, media_type = _decode_data_url(url)
            if len(data) > max_bytes:
                raise _media_error("input media exceeds the maximum size")
            return data, media_type
        current = url
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            for _ in range(6):
                validate_remote_url(current, self.allow_private_urls)
                with client.stream("GET", current, headers={"Accept": accept}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise _media_error("media URL returned an invalid redirect")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise _media_error(f"media URL returned HTTP {response.status_code}")
                    try:
                        declared = int(response.headers.get("content-length", "0") or 0)
                    except ValueError as exc:
                        raise _media_error("media URL returned an invalid Content-Length") from exc
                    if declared > max_bytes:
                        raise _media_error("input media exceeds the maximum size")
                    output = bytearray()
                    for chunk in response.iter_bytes():
                        output.extend(chunk)
                        if len(output) > max_bytes:
                            raise _media_error("input media exceeds the maximum size")
                    return bytes(output), response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
        raise _media_error("media URL redirected too many times")

    @staticmethod
    def _validate_image(data: bytes) -> tuple[Image.Image, str, str]:
        try:
            original = Image.open(io.BytesIO(data))
            image_format = original.format
            image = ImageOps.exif_transpose(original)
            image.load()
        except Exception as exc:
            raise _media_error("input is not a valid supported image") from exc
        details = ALLOWED_IMAGE_FORMATS.get(image_format or "")
        if details is None:
            raise _media_error("image format must be JPEG, PNG, WebP, HEIC or HEIF")
        _check_dimensions(*image.size, "image")
        media_type, suffix = details
        return image, media_type, suffix

    @staticmethod
    def _validate_video(data: bytes) -> tuple[str, str, int, int, dict[str, Any], Image.Image]:
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                formats = set((container.format.name or "").lower().split(","))
                if not formats.intersection({"mov", "mp4"}):
                    raise _media_error("video format must be MP4 or MOV")
                if len(container.streams.video) != 1:
                    raise _media_error("video must contain exactly one video stream")
                video = container.streams.video[0]
                codec = (video.codec_context.name or "").lower()
                if codec not in ALLOWED_VIDEO_CODECS:
                    raise _media_error("video codec must be H.264 or H.265")
                width, height = video.codec_context.width, video.codec_context.height
                _check_dimensions(width, height, "video")
                duration = _stream_duration(container, video)
                if not MEDIA_MIN_DURATION <= duration <= MEDIA_MAX_DURATION + 1e-3:
                    raise _media_error("video duration must be between 2 and 15 seconds")
                if video.average_rate is None:
                    raise _media_error("video frame rate could not be determined")
                frame_rate = float(video.average_rate)
                if not VIDEO_MIN_FPS <= frame_rate <= VIDEO_MAX_FPS:
                    raise _media_error("video frame rate must be between 23.976 and 60 FPS")
                audio_codecs = {(stream.codec_context.name or "").lower() for stream in container.streams.audio}
                if audio_codecs - ALLOWED_VIDEO_AUDIO_CODECS:
                    raise _media_error("video audio codec must be AAC or MP3")
                first_frame = next(container.decode(video=0), None)
                if first_frame is None:
                    raise _media_error("video does not contain a decodable frame")
                thumbnail = first_frame.to_image()
        except APIError:
            raise
        except Exception as exc:
            raise _media_error("input is not a valid supported video") from exc
        metadata = {
            "kind": "video",
            "duration_seconds": duration,
            "frame_rate": frame_rate,
            "video_codec": codec,
            "audio_codecs": sorted(audio_codecs),
            "has_audio": bool(audio_codecs),
        }
        return "video/mp4", ".mp4", width, height, metadata, thumbnail

    @staticmethod
    def _validate_audio(data: bytes) -> tuple[str, str, dict[str, Any]]:
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                formats = set((container.format.name or "").lower().split(","))
                if len(container.streams.audio) != 1 or container.streams.video:
                    raise _media_error("audio input must contain exactly one audio stream")
                audio = container.streams.audio[0]
                codec = (audio.codec_context.name or "").lower()
                if "mp3" in formats and codec == "mp3":
                    media_type, suffix = "audio/mpeg", ".mp3"
                elif "wav" in formats and codec.startswith("pcm_"):
                    media_type, suffix = "audio/wav", ".wav"
                else:
                    raise _media_error("audio format must be WAV or MP3")
                duration = _stream_duration(container, audio)
                if not MEDIA_MIN_DURATION <= duration <= MEDIA_MAX_DURATION + 1e-3:
                    raise _media_error("audio duration must be between 2 and 15 seconds")
        except APIError:
            raise
        except Exception as exc:
            raise _media_error("input is not a valid supported audio file") from exc
        return media_type, suffix, {"kind": "audio", "duration_seconds": duration, "audio_codec": codec}

    @staticmethod
    def _thumbnail(image: Image.Image) -> tuple[bytes, int, int]:
        thumb = image.copy()
        thumb.thumbnail((640, 640), Image.Resampling.LANCZOS)
        if thumb.mode not in {"RGB", "L"}:
            background = Image.new("RGB", thumb.size, "white")
            if "A" in thumb.getbands():
                background.paste(thumb, mask=thumb.getchannel("A"))
            else:
                background.paste(thumb)
            thumb = background
        elif thumb.mode != "RGB":
            thumb = thumb.convert("RGB")
        output = io.BytesIO()
        thumb.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue(), thumb.width, thumb.height

    def _atomic_write(self, relative_path: Path, data: bytes) -> None:
        target = self.resolve(relative_path.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
