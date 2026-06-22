from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deployment.app import (
    MANIFEST_PATH,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    _sha256,
    app,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = (
    ROOT
    / "dataset"
    / "val 23"
    / "coastal_mansion"
    / "coastalmansion701.jpg"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_selected_model_and_fixed_classes(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "efficientnet_b0_final",
        "classes": [
            "coastal_mansion",
            "dense_residential",
            "nursing_home",
            "sparse_residential",
        ],
    }


def test_web_ui_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Aerial Scene Classifier" in response.text
    assert "Run classification" in response.text


def test_checkpoint_matches_deployment_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert _sha256(MODEL_PATH) == manifest["checkpoint_sha256"]


def test_predict_returns_normalized_scores_for_an_image(client: TestClient) -> None:
    with SAMPLE_IMAGE.open("rb") as image:
        response = client.post(
            "/predict",
            files={"file": (SAMPLE_IMAGE.name, image, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_label"] in payload["scores"]
    assert payload["confidence"] == max(payload["scores"].values())
    assert set(payload["scores"]) == {
        "coastal_mansion",
        "dense_residential",
        "nursing_home",
        "sparse_residential",
    }
    assert sum(payload["scores"].values()) == pytest.approx(1.0, abs=1e-5)


def test_predict_rejects_non_image_upload(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_rejects_unreadable_image_payload(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400


def test_predict_rejects_oversized_upload(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("large.jpg", b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")},
    )

    assert response.status_code == 413
