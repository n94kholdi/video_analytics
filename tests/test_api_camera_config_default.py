from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from app.api import main
from app.geometry.config import load_camera_config


def test_empty_camera_config_upload_is_treated_as_missing() -> None:
    upload = UploadFile(file=BytesIO(), filename="", size=0)

    assert not main._has_camera_config_upload(upload)


def test_named_camera_config_upload_is_present() -> None:
    upload = UploadFile(file=BytesIO(b"camera: {}"), filename="camera.yaml", size=10)

    assert main._has_camera_config_upload(upload)


def test_bundled_example_is_copied_when_no_saved_default_exists(
    tmp_path: Path, monkeypatch,
) -> None:
    saved = tmp_path / "settings" / "camera.yaml"
    destination = tmp_path / "job" / "camera.yaml"
    destination.parent.mkdir()
    monkeypatch.setattr(main, "SAVED_CAMERA_CONFIG_PATH", saved)

    main._copy_default_camera_config(destination)

    assert load_camera_config(destination) == load_camera_config(
        main.DEFAULT_CAMERA_CONFIG_PATH
    )


def test_uploaded_config_becomes_the_saved_default(
    tmp_path: Path, monkeypatch,
) -> None:
    saved = tmp_path / "settings" / "camera.yaml"
    destination = tmp_path / "next-job" / "camera.yaml"
    destination.parent.mkdir()
    monkeypatch.setattr(main, "SAVED_CAMERA_CONFIG_PATH", saved)

    main._save_default_camera_config(main.DEFAULT_CAMERA_CONFIG_PATH)
    main._copy_default_camera_config(destination)

    assert saved.is_file()
    assert load_camera_config(destination) == load_camera_config(saved)
