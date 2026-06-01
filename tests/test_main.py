"""
tests/test_main.py
------------------
Unit tests for the Teachable Machine FastAPI backend.
Run with:  pytest tests/ -v
"""
import io
import os
import pytest
from fastapi.testclient import TestClient

# ── Import the FastAPI app ────────────────────────────────────────────────────
# Supports running from project root: pytest tests/ -v
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.main import app, sanitize_class_name

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. sanitize_class_name() utility function tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSanitizeClassName:
    def test_strips_whitespace(self):
        assert sanitize_class_name("  Cat  ") == "Cat"

    def test_replaces_spaces_with_underscore(self):
        assert sanitize_class_name("My Class") == "My_Class"

    def test_removes_special_characters(self):
        assert sanitize_class_name("Class@Name!") == "Class_Name_"

    def test_allows_hyphens_and_underscores(self):
        assert sanitize_class_name("my-class_name") == "my-class_name"

    def test_empty_string_returns_default(self):
        assert sanitize_class_name("   ") == "unnamed_class"

    def test_alphanumeric_unchanged(self):
        assert sanitize_class_name("Dog123") == "Dog123"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GET / — Health check endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthCheck:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_contains_status_key(self):
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert data["status"] == "Teachable Machine API is running"

    def test_contains_model_trained_key(self):
        response = client.get("/")
        data = response.json()
        assert "model_trained" in data
        assert isinstance(data["model_trained"], bool)

    def test_contains_active_classes_key(self):
        response = client.get("/")
        data = response.json()
        assert "active_classes" in data
        assert isinstance(data["active_classes"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GET /dataset-info — Dataset info endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetInfo:
    def test_returns_200(self):
        response = client.get("/dataset-info")
        assert response.status_code == 200

    def test_response_has_classes_key(self):
        response = client.get("/dataset-info")
        data = response.json()
        assert "classes" in data
        assert isinstance(data["classes"], dict)

    def test_all_counts_are_integers(self):
        response = client.get("/dataset-info")
        data = response.json()
        for class_name, count in data["classes"].items():
            assert isinstance(count, int), f"Count for '{class_name}' is not an integer"
            assert count >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /train — Validation: fewer than 2 classes must return 400
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainValidation:
    def test_train_with_no_classes_is_rejected(self, tmp_path, monkeypatch):
        """If dataset directory is empty, /train must return 400."""
        # Point the backend to a fresh empty dataset directory
        monkeypatch.setattr("backend.main.DATASET_DIR", str(tmp_path))
        response = client.post("/train", data={
            "backbone_name": "MobileNetV3",
            "c_value": 1.0,
            "penalty": "l2"
        })
        assert response.status_code == 400
        assert "2 distinct classes" in response.json()["detail"].lower() \
               or "no dataset" in response.json()["detail"].lower()

    def test_train_with_one_class_is_rejected(self, tmp_path, monkeypatch):
        """If only 1 class folder with images exists, /train must return 400."""
        monkeypatch.setattr("backend.main.DATASET_DIR", str(tmp_path))
        # Create one class with one dummy image
        class_dir = tmp_path / "cat"
        class_dir.mkdir()
        (class_dir / "img.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        response = client.post("/train", data={
            "backbone_name": "MobileNetV3",
            "c_value": 1.0,
            "penalty": "l2"
        })
        assert response.status_code == 400
        assert "2 distinct classes" in response.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POST /upload-sample — Basic validation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadSample:
    def test_upload_rejects_non_image_file(self):
        """Uploading a fake .jpg that is actually a text file must be rejected."""
        fake_file = io.BytesIO(b"this is not an image at all")
        response = client.post(
            "/upload-sample",
            data={"class_name": "test_class"},
            files={"files": ("fake.jpg", fake_file, "image/jpeg")}
        )
        # Should fail — either 400 (no valid images saved) or success=0
        data = response.json()
        if response.status_code == 200:
            assert data.get("saved_count", 0) == 0
        else:
            assert response.status_code == 400

    def test_upload_with_no_files_returns_400(self):
        """Calling /upload-sample with no files must return 400."""
        response = client.post(
            "/upload-sample",
            data={"class_name": "test_class"},
            files={}
        )
        assert response.status_code in (400, 422)
