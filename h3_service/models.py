from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_PROMPT_CHARS = 7000
CONCRETE_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
ALL_RATIOS = CONCRETE_RATIOS | {"adaptive"}


class APIError(Exception):
    def __init__(self, status_code: int, error_type: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


@dataclass(frozen=True)
class MediaInput:
    media_type: str
    role: str
    url: str
    source_index: int = 0


@dataclass(frozen=True)
class ParsedRequest:
    model: str
    content: list[dict[str, Any]]
    prompt: str
    mode: str
    required_capability: str
    resolution: str
    duration: int
    ratio: str
    callback_url: str | None
    media: tuple[MediaInput, ...]


def _bad(message: str) -> APIError:
    return APIError(400, "bad_request_error", message)


def parse_video_request(payload: Any) -> ParsedRequest:
    if not isinstance(payload, dict):
        raise _bad("request body must be a JSON object")
    if payload.get("model") != "MiniMax-H3":
        raise _bad("model must be MiniMax-H3")
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise _bad("content must be a non-empty array")

    text_parts: list[str] = []
    media: list[MediaInput] = []
    first_count = last_count = 0
    ref_image_count = ref_video_count = ref_audio_count = 0

    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise _bad(f"content[{index}] must be an object")
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise _bad(f"content[{index}].text must be a string")
            if text.strip():
                text_parts.append(text.strip())
            continue
        if item_type not in {"image_url", "video_url", "audio_url"}:
            raise _bad(f"content[{index}].type is unsupported")
        value = item.get(item_type)
        if not isinstance(value, dict) or not isinstance(value.get("url"), str) or not value["url"].strip():
            raise _bad(f"content[{index}].{item_type}.url is required")
        role = item.get("role")
        if item_type == "image_url":
            role = role or "first_frame"
            if role not in {"first_frame", "last_frame", "reference_image"}:
                raise _bad(f"content[{index}] has an invalid image role")
            first_count += role == "first_frame"
            last_count += role == "last_frame"
            ref_image_count += role == "reference_image"
        elif item_type == "video_url":
            if role != "reference_video":
                raise _bad(f"content[{index}] video role must be reference_video")
            ref_video_count += 1
        else:
            if role != "reference_audio":
                raise _bad(f"content[{index}] audio role must be reference_audio")
            ref_audio_count += 1
        media.append(MediaInput(item_type.removesuffix("_url"), role, value["url"].strip(), index))

    if not text_parts:
        raise _bad("content must include a non-empty text item")
    prompt = "\n".join(text_parts)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise _bad(f"combined text content must not exceed {MAX_PROMPT_CHARS} characters")
    if first_count > 1 or last_count > 1:
        raise _bad("at most one first frame and one last frame are allowed")
    if ref_image_count > 9 or ref_video_count > 3 or ref_audio_count > 3:
        raise _bad("at most 9 reference images, 3 reference videos and 3 reference audios are allowed")
    if ref_image_count + ref_video_count + ref_audio_count > 12:
        raise _bad("at most 12 reference media files are allowed")

    has_frames = first_count + last_count > 0
    has_references = ref_image_count + ref_video_count + ref_audio_count > 0
    if ref_audio_count and not (ref_image_count or ref_video_count):
        raise _bad("reference audio requires a reference image or video")

    resolution = payload.get("resolution")
    if resolution != "768P":
        raise _bad("only resolution=768P is supported by the self-hosted H3-Base service")
    duration = payload.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
        raise _bad("duration must be an integer from 4 to 15")
    ratio = payload.get("ratio", "adaptive" if (has_frames or has_references) else None)
    if ratio not in ALL_RATIOS:
        raise _bad("ratio must be one of adaptive, 21:9, 16:9, 4:3, 1:1, 3:4, 9:16")

    if has_references:
        mode, capability = "ref2va", "ref2va"
    elif has_frames:
        mode, capability, ratio = "fl2va", "fl2va", "adaptive"
    else:
        mode, capability = "t2va", "fl2va"
        if ratio not in CONCRETE_RATIOS:
            raise _bad("text-to-video requires a concrete ratio")

    callback_url = payload.get("callback_url")
    if callback_url is not None and (not isinstance(callback_url, str) or not callback_url.strip()):
        raise _bad("callback_url must be a non-empty URL")

    return ParsedRequest(
        model="MiniMax-H3",
        content=content,
        prompt=prompt,
        mode=mode,
        required_capability=capability,
        resolution=resolution,
        duration=duration,
        ratio=ratio,
        callback_url=callback_url.strip() if callback_url else None,
        media=tuple(media),
    )


def error_body(error: APIError, request_id: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": error.error_type,
            "message": error.message,
            "http_code": str(error.status_code),
        },
        "request_id": request_id,
    }
