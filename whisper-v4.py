# whisper-v3.py
import io
import os
import tempfile
from typing import Generator, Optional, List, Dict, Any

import whisper
import whisper.audio
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse

app = FastAPI(title="Whisper API - chunked")

# ajusta CORS para seu front-end (Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # ajuste em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# cache de modelos: { model_size: {"model": model_obj, "device": "cpu"/"cuda"} }
_loaded_models: Dict[str, Dict[str, Any]] = {}


def get_model(model_size: str, device: str = "cpu"):
    """
    Carrega/recupera o modelo e move para device se necessário.
    device: "cpu" ou "cuda"
    """
    key = (model_size or "base").lower()
    if key not in _loaded_models:
        _loaded_models[key] = {"model": whisper.load_model(key), "device": "cpu"}

    entry = _loaded_models[key]
    model = entry["model"]
    current = entry["device"]

    # tenta mover para CUDA se pedido e disponível
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                if current != "cuda":
                    model = model.to("cuda")
                    entry["model"] = model
                    entry["device"] = "cuda"
            else:
                # permanece em cpu
                pass
        except Exception:
            pass
    else:
        # mover para cpu se estiver em cuda
        if current != "cpu":
            model = model.to("cpu")
            entry["model"] = model
            entry["device"] = "cpu"

    return entry["model"]


def _format_timestamp(seconds: float) -> str:
    """Converte segundos (float) para 'HH:MM:SS,mmm' do SRT."""
    import math

    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def transcribe_chunked(
    file_path: str,
    model,
    language: Optional[str],
    chunk_sec: int = 300,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator que divide o arquivo em blocos de chunk_sec e para cada bloco:
      - salva bloco como WAV temporário
      - chama model.transcribe(tmp_chunk_path, **opts)
      - retorna dict com:
          {"chunk_start": <float_seconds>,
           "text": <str>,
           "segments": <list of segments from whisper (start,end,text)>}
    CLEANUP do tmp chunk é imediato após cada bloco.
    """
    # carrega áudio em 16k mono (whisper.audio usa ffmpeg)
    audio = whisper.audio.load_audio(file_path)  # numpy float32 mono 16000
    sr = whisper.audio.SAMPLE_RATE
    total = audio.shape[0]
    chunk_len = int(chunk_sec * sr)
    start_idx = 0
    chunk_index = 0

    while start_idx < total:
        end_idx = min(start_idx + chunk_len, total)
        chunk_audio = audio[start_idx:end_idx]

        # pular chunk vazio
        if chunk_audio.size == 0:
            start_idx = end_idx
            chunk_index += 1
            continue

        chunk_start_seconds = start_idx / sr

        # salva chunk em WAV temporário
        tmp_chunk_fd, tmp_chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_chunk_fd)
        # soundfile.write
        sf.write(tmp_chunk_path, chunk_audio, sr)

        opts = {"task": "transcribe", "verbose": False}
        if language:
            opts["language"] = language

        try:
            # opção segura: passar caminho do arquivo
            result = model.transcribe(tmp_chunk_path, **opts)
            text = (result.get("text") or "").strip()
            segments = result.get("segments") or []
            # emissao: segmentos retornados tem timestamps relativos ao chunk
            yield {"chunk_start": chunk_start_seconds, "text": text, "segments": segments}
        except Exception as e:
            yield {"chunk_start": chunk_start_seconds, "text": f"[ERRO] {e}", "segments": []}
        finally:
            # remover o arquivo do chunk
            try:
                if os.path.exists(tmp_chunk_path):
                    os.remove(tmp_chunk_path)
            except Exception:
                pass

        start_idx = end_idx
        chunk_index += 1


def build_srt(all_segments: List[Dict[str, Any]]) -> str:
    """
    all_segments: lista de segmentos com start/end/text já ajustados para tempo absoluto.
    Gera SRT ordenado por start.
    """
    # ordenar
    segs = sorted(all_segments, key=lambda s: s["start"])
    lines = []
    for i, s in enumerate(segs, start=1):
        start_ts = _format_timestamp(s["start"])
        end_ts = _format_timestamp(s["end"])
        text = (s.get("text") or "").strip()
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")  # blank line
    return "\n".join(lines)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    output_format: str = Form("txt"),  # "txt" ou "srt"
    model_size: str = Form("base"),
    language: Optional[str] = Form(None),
    device: str = Form("cpu"),  # "cpu" ou "cuda"
    chunk_sec: int = Form(300),  # seconds per chunk (default 5min)
):
    """
    Recebe upload e:
      - se output_format == "txt": retorna streaming do texto (chunk por chunk)
      - se output_format == "srt": processa todos os chunks, junta segmentos com offsets e retorna .srt
    """
    # salva upload em arquivo temporário (manter até o final do generator)
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_fd, src_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    try:
        content = await file.read()
        with open(src_path, "wb") as f:
            f.write(content)
    except Exception as e:
        # cleanup se falhar
        try:
            if os.path.exists(src_path):
                os.remove(src_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Falha ao salvar upload: {e}")

    # carregar modelo (pode demorar se não estiver em cache)
    model = get_model(model_size, device=device)

    base_name = os.path.splitext(file.filename or "transcript")[0]

    # Modo SRT: coletar todos segments ajustando tempo absoluto
    if output_format.lower() == "srt":
        all_segments = []
        try:
            for chunk_info in transcribe_chunked(src_path, model, language, chunk_sec=chunk_sec):
                # chunk_info contains 'chunk_start' and 'segments' from whisper (relative)
                chunk_start = chunk_info.get("chunk_start", 0.0)
                segments = chunk_info.get("segments", [])
                # ajustar timestamps dos segmentos
                for seg in segments:
                    start_abs = (seg.get("start", 0.0) or 0.0) + chunk_start
                    end_abs = (seg.get("end", 0.0) or 0.0) + chunk_start
                    text = seg.get("text", "") or ""
                    all_segments.append({"start": start_abs, "end": end_abs, "text": text})
            # após processar todos chunks, gerar SRT
            srt_text = build_srt(all_segments)
        finally:
            # remover arquivo fonte temporário
            try:
                if os.path.exists(src_path):
                    os.remove(src_path)
            except Exception:
                pass

        return StreamingResponse(
            io.StringIO(srt_text),
            media_type="application/x-subrip",
            headers={"Content-Disposition": f'attachment; filename="{base_name}.srt"'},
        )

    # Modo TXT: streaming em tempo real (bloco por bloco)
    def stream_txt():
        try:
            for chunk_info in transcribe_chunked(src_path, model, language, chunk_sec=chunk_sec):
                text = chunk_info.get("text", "") or ""
                # mantém streaming contínuo; cada chunk termina com newline
                yield (text + "\n").encode("utf-8")
        finally:
            # limpar arquivo fonte quando stream terminar/encerrar
            try:
                if os.path.exists(src_path):
                    os.remove(src_path)
            except Exception:
                pass

    return StreamingResponse(
        stream_txt(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base_name}.txt"'},
    )
