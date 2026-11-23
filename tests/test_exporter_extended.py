"""Comprehensive unit and edge-case tests for librarian.exporter."""

from __future__ import annotations

import io
import os
import platform
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from librarian import exporter


def test_detect_video_path_from_html_via_explicit_url(tmp_path: Path):
    unit_dir = tmp_path / "wiki" / "Test_Unit"
    media_dir = unit_dir / "sources" / "media"
    handouts_dir = unit_dir / "handouts"
    media_dir.mkdir(parents=True)
    handouts_dir.mkdir(parents=True)

    video_file = media_dir / "sample_lecture.mp4"
    video_file.write_bytes(b"mock video data")

    html_path = handouts_dir / "Test_Unit_video_quiz.html"
    html_content = f"""
    <!DOCTYPE html><html><body>
    <script>
    const quizData = {{
        "title": "Video Quiz",
        "video_url": "wiki/Test_Unit/sources/media/{video_file.name}"
    }};
    </script></body></html>
    """
    html_path.write_text(html_content, encoding="utf-8")

    with patch.object(exporter.config, "project_root", tmp_path):
        detected = exporter.detect_video_path_from_html(html_content, html_path)
        assert detected is not None
        assert detected.name == "sample_lecture.mp4"


def test_detect_video_path_from_html_via_stem_fallback(tmp_path: Path):
    unit_dir = tmp_path / "wiki" / "Test_Unit"
    media_dir = unit_dir / "sources" / "media"
    handouts_dir = unit_dir / "handouts"
    media_dir.mkdir(parents=True)
    handouts_dir.mkdir(parents=True)

    media_file = media_dir / "lecture_01.webm"
    media_file.write_bytes(b"mock webm data")

    html_path = handouts_dir / "lecture_01_video_quiz.html"
    html_content = "<!DOCTYPE html><html><body>No video_url here</body></html>"
    html_path.write_text(html_content, encoding="utf-8")

    with patch.object(exporter.config, "project_root", tmp_path):
        detected = exporter.detect_video_path_from_html(html_content, html_path)
        assert detected is not None
        assert detected.name == "lecture_01.webm"


def test_make_portable_html_content():
    raw_html = 'const quizData = {"video_url": "/wiki/Unit1/sources/media/clip.mp4", "count": 5};'
    updated = exporter.make_portable_html_content(raw_html, "clip.mp4")
    assert '"video_url": "clip.mp4"' in updated

    # None media_filename returns original untouched
    assert exporter.make_portable_html_content(raw_html, None) == raw_html


def test_export_portable_bundle_missing_html():
    res = exporter.export_portable_bundle("/non/existent/path/quiz.html")
    assert res is False


def test_export_portable_bundle_zip_mode(tmp_path: Path):
    unit_dir = tmp_path / "wiki" / "Unit_Video"
    handouts_dir = unit_dir / "handouts"
    media_dir = unit_dir / "sources" / "media"
    handouts_dir.mkdir(parents=True)
    media_dir.mkdir(parents=True)

    video = media_dir / "talk.mp4"
    video.write_bytes(b"video stream 12345")

    html_path = handouts_dir / "Unit_Video_video_quiz.html"
    html_content = f'const quizData = {{"video_url": "wiki/Unit_Video/sources/media/{video.name}"}};'
    html_path.write_text(html_content, encoding="utf-8")

    out_zip = tmp_path / "custom_bundle.zip"

    with patch.object(exporter.config, "project_root", tmp_path):
        bundle = exporter.export_portable_bundle(html_path, video_path=video, output_path=out_zip)
        assert bundle == out_zip
        assert out_zip.exists()

        with zipfile.ZipFile(out_zip, "r") as zf:
            namelist = zf.namelist()
            assert "Unit_Video_video_quiz.html" in namelist
            assert "talk.mp4" in namelist


def test_export_portable_bundle_dir_mode(tmp_path: Path):
    unit_dir = tmp_path / "wiki" / "Unit_Video"
    handouts_dir = unit_dir / "handouts"
    media_dir = unit_dir / "sources" / "media"
    handouts_dir.mkdir(parents=True)
    media_dir.mkdir(parents=True)

    video = media_dir / "talk.mp4"
    video.write_bytes(b"video stream 12345")

    html_path = handouts_dir / "Unit_Video_video_quiz.html"
    html_content = 'const quizData = {"video_url": "talk.mp4"};'
    html_path.write_text(html_content, encoding="utf-8")

    out_dir = tmp_path / "extracted_portable"

    with patch.object(exporter.config, "project_root", tmp_path):
        res_file = exporter.export_portable_bundle(html_path, video_path=video, output_path=out_dir)
        assert res_file == out_dir / "Unit_Video_video_quiz.html"
        assert res_file.exists()
        assert (out_dir / "talk.mp4").exists()


def test_open_directory_in_explorer_nonexistent():
    assert exporter.open_directory_in_explorer("/path/that/does/not/exist/at/all") is False


def test_open_directory_in_explorer_linux_xdg(tmp_path: Path):
    with patch("platform.system", return_value="Linux"), \
         patch("os.environ.get", return_value=None), \
         patch("shutil.which", side_effect=lambda cmd: "/usr/bin/xdg-open" if cmd == "xdg-open" else None), \
         patch("subprocess.Popen") as mock_popen:
        ok = exporter.open_directory_in_explorer(tmp_path)
        assert ok is True
        mock_popen.assert_called_once()
