from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO = REPO_ROOT / "Aca-feedback-buurt-check1.ogg"
DEFAULT_OUTPUT = REPO_ROOT / "Aca-feedback-buurt-check1.txt"
DEFAULT_LANGUAGE = "sr"
DEFAULT_MODEL_SIZE = "base"
DEFAULT_QUEUE_SECONDS = 20

MODEL_FILES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}

MODEL_URL_TEMPLATE = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe Aca-feedback-buurt-check1.ogg with ffmpeg's whisper.cpp "
            "filter and save the transcript as a text file."
        )
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help=f"Audio file to transcribe. Default: {DEFAULT_AUDIO.name}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Transcript file to write. Default: {DEFAULT_OUTPUT.name}",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Whisper language code. Default: {DEFAULT_LANGUAGE}",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_FILES),
        default=DEFAULT_MODEL_SIZE,
        help=f"whisper.cpp model size to download/use. Default: {DEFAULT_MODEL_SIZE}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Use an existing whisper.cpp model file instead of downloading one.",
    )
    parser.add_argument(
        "--queue-seconds",
        type=int,
        default=DEFAULT_QUEUE_SECONDS,
        help=f"Audio chunk size passed to ffmpeg whisper. Default: {DEFAULT_QUEUE_SECONDS}",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Ask ffmpeg whisper to use GPU. CPU is the safe default.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path to the ffmpeg executable. Default: ffmpeg from PATH.",
    )
    return parser.parse_args()


def default_model_dir() -> Path:
    return REPO_ROOT / ".codex_tmp" / "whisper-models"


def ensure_ffmpeg(ffmpeg: str) -> str:
    resolved = shutil.which(ffmpeg)
    if resolved:
        return resolved
    candidate = Path(ffmpeg)
    if candidate.is_file():
        return str(candidate)
    raise SystemExit(
        "ffmpeg was not found. Install ffmpeg or pass --ffmpeg with its full path."
    )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temp_path.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                percent = downloaded / total_bytes * 100
                print(
                    f"\rDownloading model: {percent:5.1f}% "
                    f"({downloaded / 1024 / 1024:.1f} / {total_bytes / 1024 / 1024:.1f} MB)",
                    end="",
                    flush=True,
                )
    temp_path.replace(destination)
    if total_bytes:
        print()


def ensure_model(model_size: str, model_path: Path | None) -> Path:
    if model_path:
        if not model_path.is_file():
            raise SystemExit(f"Model file does not exist: {model_path}")
        return model_path.resolve()

    filename = MODEL_FILES[model_size]
    destination = default_model_dir() / filename
    if destination.is_file():
        return destination

    url = MODEL_URL_TEMPLATE.format(filename=filename)
    print(f"Downloading whisper.cpp model '{model_size}' to {destination}")
    download_file(url, destination)
    return destination


def ffmpeg_filter_escape(value: str) -> str:
    return (
        value.replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


def path_for_filter(path: Path, ffmpeg_cwd: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ffmpeg_cwd.resolve())
        return ffmpeg_filter_escape(relative.as_posix())
    except ValueError:
        try:
            relative = Path(os.path.relpath(resolved, ffmpeg_cwd))
            return ffmpeg_filter_escape(relative.as_posix())
        except ValueError:
            return ffmpeg_filter_escape(resolved.as_posix())


def build_whisper_filter(
    model_path: Path,
    language: str,
    queue_seconds: int,
    output_path: Path,
    use_gpu: bool,
    ffmpeg_cwd: Path,
) -> str:
    return (
        "whisper="
        f"model={path_for_filter(model_path, ffmpeg_cwd)}:"
        f"language={ffmpeg_filter_escape(language)}:"
        f"queue={queue_seconds}:"
        f"use_gpu={'true' if use_gpu else 'false'}:"
        f"destination={path_for_filter(output_path, ffmpeg_cwd)}:"
        "format=text"
    )


def run_transcription(
    ffmpeg_bin: str,
    audio_path: Path,
    output_path: Path,
    filter_spec: str,
    ffmpeg_cwd: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(audio_path.resolve()),
        "-af",
        filter_spec,
        "-f",
        "null",
        os.devnull,
    ]
    print(f"Running ffmpeg transcription for {audio_path.name}")
    subprocess.run(command, check=True, cwd=ffmpeg_cwd)

    if not output_path.is_file():
        raise SystemExit(f"ffmpeg completed but did not create {output_path}")
    if not output_path.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Transcript file is empty: {output_path}")


def main() -> None:
    args = parse_args()
    audio_path = args.audio.resolve()
    output_path = args.output.resolve()

    if not audio_path.is_file():
        raise SystemExit(f"Audio file does not exist: {audio_path}")
    if args.queue_seconds <= 0:
        raise SystemExit("--queue-seconds must be greater than 0")

    ffmpeg_bin = ensure_ffmpeg(args.ffmpeg)
    model_path = ensure_model(args.model_size, args.model_path)
    filter_spec = build_whisper_filter(
        model_path=model_path,
        language=args.language,
        queue_seconds=args.queue_seconds,
        output_path=output_path,
        use_gpu=args.use_gpu,
        ffmpeg_cwd=REPO_ROOT,
    )
    run_transcription(
        ffmpeg_bin=ffmpeg_bin,
        audio_path=audio_path,
        output_path=output_path,
        filter_spec=filter_spec,
        ffmpeg_cwd=REPO_ROOT,
    )
    print(f"Transcript written to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg transcription failed with exit code {exc.returncode}")
