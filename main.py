import io
import os
import tempfile
from typing import Generator, Optional, List, Dict, Any

import whisper
import whisper.audio
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Whisper API - chunked")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

_loaded_models: Dict[str, Dict[str, Any]] = {}

def get_model(model_size: str, device: str = "cpu"):
    _loaded_models.clear()
    key = (model_size or "base").lower()
    model = whisper.load_model(key)

    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                model = model.to("cuda")
                device = "cuda"
        except Exception:
            device = "cpu"

    _loaded_models[key] = {"model": model, "device": device}
    return model

def _format_timestamp(seconds: float) -> str:
    import math
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def transcribe_chunked(file_path: str, model, language: Optional[str], chunk_sec: int = 300) -> Generator[Dict[str, Any], None, None]:
    audio = whisper.audio.load_audio(file_path)
    sr = whisper.audio.SAMPLE_RATE
    total = audio.shape[0]
    chunk_len = int(chunk_sec * sr)
    start_idx = 0

    while start_idx < total:
        end_idx = min(start_idx + chunk_len, total)
        chunk_audio = audio[start_idx:end_idx]

        if chunk_audio.size == 0:
            start_idx = end_idx
            continue

        chunk_start_seconds = start_idx / sr
        tmp_chunk_fd, tmp_chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_chunk_fd)
        sf.write(tmp_chunk_path, chunk_audio, sr)

        opts = {"task": "transcribe", "verbose": False}
        if language:
            opts["language"] = language

        try:
            result = model.transcribe(tmp_chunk_path, **opts)
            text = (result.get("text") or "").strip()
            segments = result.get("segments") or []
            yield {"chunk_start": chunk_start_seconds, "text": text, "segments": segments}
        except Exception as e:
            yield {"chunk_start": chunk_start_seconds, "text": f"[ERRO] {e}", "segments": []}
        finally:
            if os.path.exists(tmp_chunk_path):
                os.remove(tmp_chunk_path)

        start_idx = end_idx

def build_srt(all_segments: List[Dict[str, Any]]) -> str:
    segs = sorted(all_segments, key=lambda s: s["start"])
    lines = []
    for i, s in enumerate(segs, start=1):
        start_ts = _format_timestamp(s["start"])
        end_ts = _format_timestamp(s["end"])
        text = (s.get("text") or "").strip()
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    output_format: str = Form("txt"),
    model_size: str = Form("base"),
    language: Optional[str] = Form(None),
    device: str = Form("cpu"),
    chunk_sec: int = Form(300),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_fd, src_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    try:
        content = await file.read()
        with open(src_path, "wb") as f:
            f.write(content)
    except Exception as e:
        if os.path.exists(src_path):
            os.remove(src_path)
        raise HTTPException(status_code=500, detail=f"Falha ao salvar upload: {e}")

    model = get_model(model_size, device=device)
    base_name = os.path.splitext(file.filename or "transcript")[0]

    if output_format.lower() == "srt":
        all_segments = []
        try:
            for chunk_info in transcribe_chunked(src_path, model, language, chunk_sec=chunk_sec):
                chunk_start = chunk_info.get("chunk_start", 0.0)
                segments = chunk_info.get("segments", [])
                for seg in segments:
                    start_abs = seg.get("start", 0.0) + chunk_start
                    end_abs = seg.get("end", 0.0) + chunk_start
                    text = seg.get("text", "") or ""
                    all_segments.append({"start": start_abs, "end": end_abs, "text": text})
            srt_text = build_srt(all_segments)
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)

        return StreamingResponse(
            io.StringIO(srt_text),
            media_type="application/x-subrip",
            headers={"Content-Disposition": f'attachment; filename="{base_name}.srt"'},
        )

    def stream_txt():
        try:
            for chunk_info in transcribe_chunked(src_path, model, language, chunk_sec=chunk_sec):
                text = chunk_info.get("text", "") or ""
                yield (text + "\n").encode("utf-8")
        finally:
            if os.path.exists(src_path):
                os.remove(src_path)

    return StreamingResponse(
        stream_txt(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base_name}.txt"'},
    )
