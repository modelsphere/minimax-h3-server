import httpx
import pytest
from fastapi.testclient import TestClient

from h3_router.server import BackendPool, RouterSettings, _is_ref_request, create_app


def test_router_classifies_reference_content():
    assert _is_ref_request(b'{"content":[{"type":"image_url","role":"reference_image"}]}')
    assert not _is_ref_request(b'{"content":[{"type":"image_url","role":"first_frame"}]}')


def test_pool_rotates_without_persistence():
    pool = BackendPool(("http://a", "http://b"))
    assert pool.candidates() == ["http://a", "http://b"]
    assert pool.candidates() == ["http://b", "http://a"]


def test_ref_returns_501_without_target():
    app = create_app(RouterSettings(("http://fl",), ()))
    with TestClient(app) as client:
        response = client.post("/v2/video_generation", json={"content": [{"type": "image_url", "role": "reference_image"}]})
    assert response.status_code == 501
    assert response.json()["error"]["type"] == "not_implemented_error"


@pytest.mark.parametrize("frame_roles", [[], ["first_frame", "last_frame"]])
def test_ref_create_is_forwarded_to_ref2va_pool(monkeypatch, frame_roles):
    requests = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def build_request(self, method, url, headers, content):
            request = httpx.Request(method, url, headers=headers, content=content)
            requests.append(request)
            return request

        async def send(self, request, stream):
            return httpx.Response(200, stream=httpx.ByteStream(b'{"task_id":"ref-task"}'), request=request)

        async def aclose(self):
            pass

    monkeypatch.setattr("h3_router.server.httpx.AsyncClient", FakeClient)
    app = create_app(RouterSettings(("http://fl:8000",), ("http://ref:8000",)))
    with TestClient(app) as client:
        response = client.post(
            "/v2/video_generation",
            json={"content": [
                *[{"type": "image_url", "role": role} for role in frame_roles],
                {"type": "video_url", "role": "reference_video"},
            ]},
        )

    assert response.status_code == 200
    assert str(requests[0].url) == "http://ref:8000/v2/video_generation"


def test_non_create_requests_use_shared_pool_and_forward_public_origin(monkeypatch):
    requests = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def build_request(self, method, url, headers, content):
            request = httpx.Request(method, url, headers=headers, content=content)
            requests.append(request)
            return request

        async def send(self, request, stream):
            return httpx.Response(200, stream=httpx.ByteStream(b"ok"), request=request)

        async def aclose(self):
            pass

    monkeypatch.setattr("h3_router.server.httpx.AsyncClient", FakeClient)
    app = create_app(RouterSettings(("http://fl-a:8000",), ("http://ref-a:8000",)))
    with TestClient(app) as client:
        first = client.get(
            "/v2/query/video_generation/task-1",
            headers={"x-forwarded-proto": "https", "x-forwarded-host": "router.example"},
        )
        second = client.get("/v2/query/video_generation/task-1")

    assert first.status_code == 200
    assert second.status_code == 200
    assert str(requests[0].url) == "http://fl-a:8000/v2/query/video_generation/task-1"
    assert str(requests[1].url) == "http://ref-a:8000/v2/query/video_generation/task-1"
    assert requests[0].headers["x-forwarded-proto"] == "https"
    assert requests[0].headers["x-forwarded-host"] == "router.example"


def test_shared_pool_deduplicates_urls(monkeypatch):
    requests = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def build_request(self, method, url, headers, content):
            request = httpx.Request(method, url, headers=headers, content=content)
            requests.append(request)
            return request

        async def send(self, request, stream):
            return httpx.Response(200, stream=httpx.ByteStream(b"ok"), request=request)

        async def aclose(self):
            pass

    monkeypatch.setattr("h3_router.server.httpx.AsyncClient", FakeClient)
    app = create_app(RouterSettings(("http://shared:8000",), ("http://shared:8000",)))
    with TestClient(app) as client:
        client.get("/dashboard")
        client.get("/dashboard")

    assert [str(request.url) for request in requests] == [
        "http://shared:8000/dashboard",
        "http://shared:8000/dashboard",
    ]
