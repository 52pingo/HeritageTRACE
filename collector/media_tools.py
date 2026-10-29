"""Optional OCR and speech-to-text helpers, imported only when enabled."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from config import Settings


log = logging.getLogger("media_tools")


def ocr_image_url(client: Any, url: str, settings: Settings, languages: str = "chi_sim+eng") -> str:
    if not settings.ocr_enabled or not url:
        return ""
    try:
        from PIL import Image
        import pytesseract

        response = client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".img") as handle:
            handle.write(response.content)
            handle.flush()
            with Image.open(handle.name) as image:
                return pytesseract.image_to_string(image, lang=languages).strip()
    except Exception as exc:
        log.warning("OCR 失败 %s: %s", url, exc)
        return ""


def whisper_transcribe_url(url: str, settings: Settings) -> str:
    if not settings.whisper_enabled or not url:
        return ""
    try:
        import whisper
        import yt_dlp

        with tempfile.TemporaryDirectory(prefix="social-deep-") as temp:
            template = str(Path(temp) / "audio.%(ext)s")
            options = {
                "format": "bestaudio/best",
                "outtmpl": template,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            }
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.download([url])
            audio = next(Path(temp).glob("audio.*"))
            model = whisper.load_model(settings.whisper_model)
            result = model.transcribe(str(audio))
            return str(result.get("text", "")).strip()
    except Exception as exc:
        log.warning("Whisper 转写失败 %s: %s", url, exc)
        return ""

