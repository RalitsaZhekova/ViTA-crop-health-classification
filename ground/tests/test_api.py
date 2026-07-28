from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from prithvi_ground.api import create_app
from test_catalog import _build_bundle


def _upload(client: TestClient, bundle: Path, *, api_key: str | None = None):
    headers = {"X-API-Key": api_key} if api_key is not None else {}
    with (
        (bundle / "scene.json").open("rb") as scene_json,
        (bundle / "scene.webp").open("rb") as scene_webp,
        (bundle / "condition.png").open("rb") as condition_png,
    ):
        return client.post(
            "/api/v1/scenes",
            headers=headers,
            files={
                "scene_json": ("scene.json", scene_json, "application/json"),
                "scene_webp": ("scene.webp", scene_webp, "image/webp"),
                "condition_png": ("condition.png", condition_png, "image/png"),
            },
        )


def test_health_and_empty_collections(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "store")) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.headers["x-content-type-options"] == "nosniff"
        assert client.get("/api/v1/scenes").json()["items"] == []
        assert client.get("/api/v1/regions").json()["items"] == []


def test_professional_web_client_is_served_without_external_dependencies(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path / "store")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "default-src 'self'" in page.headers["content-security-policy"]
        assert 'id="scene-viewer"' in page.text
        assert 'id="condition-score"' in page.text
        assert 'id="cloud-bar"' in page.text
        assert "http://" not in page.text
        assert "https://" not in page.text

        stylesheet = client.get("/static/styles.css")
        script = client.get("/static/app.js")
        assert stylesheet.status_code == 200
        assert stylesheet.headers["content-type"].startswith("text/css")
        assert "@media (max-width: 620px)" in stylesheet.text
        assert script.status_code == 200
        assert "application/javascript" in script.headers["content-type"]
        assert "/cells/" in script.text
        assert "pointermove" in script.text
        assert "renderHistory" in script.text
        assert '"cvi", "CVI"' in script.text
        assert '"rgb_brightness", "RGB"' in script.text


def test_upload_and_complete_scene_api(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    with TestClient(create_app(tmp_path / "store")) as client:
        uploaded = _upload(client, bundle)
        assert uploaded.status_code == 201
        assert uploaded.json()["created"] is True

        repeated = _upload(client, bundle)
        assert repeated.status_code == 200
        assert repeated.json()["created"] is False

        listing = client.get("/api/v1/scenes").json()
        assert listing["count"] == 1
        assert listing["items"][0]["links"]["preview"].endswith("/preview")

        scene = client.get("/api/v1/scenes/sentinel-scene-01")
        assert scene.status_code == 200
        assert scene.json()["condition"]["score"] == 68.5

        manifest = client.get("/api/v1/scenes/sentinel-scene-01/manifest").json()["scene"]
        assert manifest["interaction_grid"]["cells"][0]["id"] == "r00c00"

        cell = client.get("/api/v1/scenes/sentinel-scene-01/cells/r00c00")
        assert cell.status_code == 200
        assert cell.json()["cell"]["condition_score"] == 68.5

        assert client.get("/api/v1/regions").json()["count"] == 1
        assert client.get("/api/v1/regions/demo-region/latest").status_code == 200
        history = client.get("/api/v1/regions/demo-region/history").json()
        assert history["count"] == 1


def test_assets_have_immutable_cache_and_conditional_get(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    with TestClient(create_app(tmp_path / "store")) as client:
        _upload(client, bundle)
        preview = client.get("/api/v1/scenes/sentinel-scene-01/preview")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/webp"
        assert "immutable" in preview.headers["cache-control"]
        etag = preview.headers["etag"]
        cached = client.get(
            "/api/v1/scenes/sentinel-scene-01/preview",
            headers={"If-None-Match": etag},
        )
        assert cached.status_code == 304
        assert cached.content == b""

        overlay = client.get("/api/v1/scenes/sentinel-scene-01/condition-overlay")
        assert overlay.status_code == 200
        assert overlay.headers["content-type"] == "image/png"


def test_upload_auth_limit_validation_and_not_found(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    app = create_app(tmp_path / "store", upload_api_key="secret", max_bundle_bytes=32)
    with TestClient(app) as client:
        assert _upload(client, bundle).status_code == 401
        assert _upload(client, bundle, api_key="wrong").status_code == 401
        too_large = _upload(client, bundle, api_key="secret")
        assert too_large.status_code == 413

        assert client.get("/api/v1/scenes/missing").status_code == 404
        assert client.get("/api/v1/regions/missing/history").status_code == 404


def test_upload_rejects_wrong_filename_and_corruption(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    with TestClient(create_app(tmp_path / "store")) as client:
        with (
            (bundle / "scene.json").open("rb") as scene_json,
            (bundle / "scene.webp").open("rb") as scene_webp,
            (bundle / "condition.png").open("rb") as condition_png,
        ):
            response = client.post(
                "/api/v1/scenes",
                files={
                    "scene_json": ("wrong.json", scene_json, "application/json"),
                    "scene_webp": ("scene.webp", scene_webp, "image/webp"),
                    "condition_png": ("condition.png", condition_png, "image/png"),
                },
            )
        assert response.status_code == 422

        (bundle / "condition.png").write_bytes(b"corrupt")
        corrupted = _upload(client, bundle)
        assert corrupted.status_code == 422
        assert "Byte count" in corrupted.json()["detail"]
