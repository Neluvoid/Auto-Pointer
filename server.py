"""
server.py
=========
フェーズ2: WebSocketサーバー (FastAPI + python-socketio) + リアルタイムASR統合

機能:
  - スライドJSONとスライド画像を読み込んでフロントエンドに配信
  - フロントエンドからASRテキストを受信し、semantic_pointer で意味理解を実行
  - ポインティング結果をリアルタイムでフロントエンドにブロードキャスト
  - スライド切り替えイベントの処理
  - マイク入力 → faster-whisper → Socket.IO のリアルタイムASRパイプライン (タスク2.3)

起動方法:
    # 基本（手動テキスト入力のみ）
    python server.py --slide-json slide_data.json --images-dir slides_images

    # マイクASR有効化（--asr フラグを追加）
    python server.py --slide-json slide_data.json --images-dir slides_images --asr

    # PPTXから直接起動
    python server.py --pptx your_slide.pptx --asr

    # ASRモデルサイズ指定（デフォルト: small）
    python server.py --slide-json slide_data.json --images-dir slides_images --asr --asr-model base

APIエンドポイント:
    GET  /                    → index.html (フロントエンド)
    GET  /api/slides          → スライドメタデータ一覧
    GET  /api/slide/{n}/image → スライド画像（PNG）
    GET  /api/slide/{n}/elements → スライド要素のJSON
    GET  /api/asr/status      → ASRスレッドの稼働状態

Socket.IOイベント:
    クライアント → サーバー:
        asr_text      : { text: "発話テキスト", slide_num: 1 }  ← 手動入力
        change_slide  : { slide_num: 1 }
        asr_control   : { action: "start" | "stop" }            ← マイクON/OFF

    サーバー → クライアント:
        pointing_result : ポインティング結果 (semantic_pointer.py の返り値)
        slide_changed   : { slide_num: 1 }
        status          : { message: "処理中..." }
        asr_transcript  : { text: "認識テキスト", confidence: -0.3 }  ← 字幕表示用
        asr_status      : { running: true|false }
        error           : { message: "エラーメッセージ" }
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP二重ロード回避
import sys
import threading
import time
import traceback
import collections
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import socketio
from contextlib import asynccontextmanager

# semantic_pointer を同じディレクトリから読み込む
sys.path.insert(0, str(Path(__file__).parent))
try:
    from semantic_pointer import (
        load_slide_data, get_elements_for_slide, get_pointing_target, get_chart_pointing
    )
except ImportError:
    sys.exit("[ERROR] semantic_pointer.py が見つかりません。同じディレクトリに置いてください。")

# ============================================================
# ロガー設定
# ============================================================

def _setup_logger() -> logging.Logger:
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"server_{timestamp}.log"
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root = logging.getLogger("autopointer")
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)
    root.addHandler(fh)
    root.info(f"ログ出力先: {log_file}")
    return root

logger = _setup_logger()

# ============================================================
# グローバル状態
# ============================================================

_slide_data: dict = {}          # slide_parser.py の出力JSON
_images_dir: str  = ""          # スライド画像ディレクトリ
_current_slide: int = 1         # 現在のスライド番号（1-indexed）
_llm_model: str = "llama3"      # 使用するOllamaモデル（ポインティング推論用）
_asr_prompt_model: str = "llama3"  # ASRプロンプト生成用モデル（精度重視、遅くてもOK）
_processing: bool = False       # LLM処理中フラグ（二重実行防止）

# --- ASR関連 ---
_asr_enabled: bool = False      # --asr フラグでTrue（サーバー側マイク）
_browser_asr_model = None       # ブラウザASR用Whisperモデル（遅延ロード）
_browser_asr_lock = threading.Lock()  # Whisperモデルのスレッドセーフアクセス用
_asr_thread: threading.Thread | None = None
_asr_stop_event: threading.Event = threading.Event()
_asr_running: bool = False      # 現在マイク収録中かどうか
_asr_model_size: str = "small"  # Whisperモデルサイズ
_asr_initial_prompt: str = "プレゼンテーション スライド 発表"  # 動的更新される
_asr_prompt_cache: dict[int, str] = {}   # slide_num → prompt のキャッシュ
_asr_prompt_ready: set[int] = set()       # 生成完了済みスライド番号

# asyncio イベントループへの参照（スレッドからemitするために必要）
_main_loop: asyncio.AbstractEventLoop | None = None

# --- スロットリング・キュー（タスク2.4）---
_COOLDOWN_SEC: float = 2.0          # 同一テキスト連続送信の無視時間（秒）
_last_asr_text: str = ""            # 直前の発話テキスト
_last_asr_time: float = 0.0         # 直前の発話時刻
_pending_text: str | None = None    # LLM処理中に届いた発話（1件のみ保持）

# --- タイミング計測（タスク2.5）---
_timing_log: list[dict] = []        # 計測結果リスト（最大100件）

# --- PPTXアップロード状態 ---
_pptx_setup_status: dict = {"state": "idle", "message": "", "progress": 0}
# state: idle / processing / done / error

# ============================================================
# FastAPI + Socket.IO セットアップ
# ============================================================

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

@asynccontextmanager
async def lifespan(application: FastAPI):
    """サーバー起動・終了時の処理。"""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("[INFO] asyncioループ取得完了")
    # ASR prompt生成をバックグラウンドスレッドで開始（サーバー起動をブロックしない）
    if _slide_data:
        t = threading.Thread(target=_generate_all_asr_prompts, daemon=True, name="asr-prompt-gen")
        t.start()
    if _asr_enabled:
        logger.info("[INFO] マイクASRを自動起動します...")
        _start_asr_thread()
    yield
    if _asr_running:
        _stop_asr_thread()

app = FastAPI(title="Auto-Pointing Presentation Server", lifespan=lifespan)

# Socket.IO は FastAPI をラップする形でトップレベルに配置する。
# app.mount() を使うとStarletteのWSルーティングが干渉してエラーになるため、
# socketio.ASGIApp(sio, other_asgi_app=app) でラップし、
# uvicorn には combined_app を渡す。
# Socket.IOの接続パスは /socket.io (デフォルト)
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

# static ディレクトリがあればマウント（CSS/JS等）
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# REST APIエンドポイント
# ============================================================

@app.get("/")
async def index():
    """フロントエンドHTMLを返す。"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>index.html が見つかりません</h1>", status_code=404)


@app.get("/api/slides")
async def get_slides():
    """スライドメタデータ一覧を返す。"""
    if not _slide_data:
        raise HTTPException(status_code=503, detail="スライドデータ未読み込み")
    slides = _slide_data.get("slides", [])
    return {
        "total": len(slides),
        "slide_width_pt": _slide_data.get("slide_width_pt"),
        "slide_height_pt": _slide_data.get("slide_height_pt"),
        "slides": [
            {
                "slide_number": s["slide_number"],
                "element_count": len(s["elements"]),
            }
            for s in slides
        ],
    }


@app.get("/api/slide/{slide_num}/image")
async def get_slide_image(slide_num: int):
    """スライド画像（PNG）を返す。"""
    image_path = _get_image_path(slide_num)
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"スライド {slide_num} の画像が見つかりません")
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/asr/status")
async def get_asr_status():
    """ASRスレッドの稼働状態を返す。"""
    return {
        "enabled": _asr_enabled,
        "running": _asr_running,
        "model":   _asr_model_size,
    }


@app.get("/api/timing/report")
async def get_timing_report():
    """タイミング計測ログをJSON/CSV形式で返す（論文用データ収集）。"""
    fmt = "json"
    if not _timing_log:
        return {"count": 0, "data": []}

    asr_times   = [r["asr_sec"]   for r in _timing_log]
    llm_times   = [r["llm_sec"]   for r in _timing_log]
    total_times = [r["total_sec"] for r in _timing_log]

    return {
        "count": len(_timing_log),
        "summary": {
            "asr_avg":   round(sum(asr_times)   / len(asr_times),   3),
            "llm_avg":   round(sum(llm_times)   / len(llm_times),   3),
            "total_avg": round(sum(total_times) / len(total_times), 3),
            "total_min": round(min(total_times), 3),
            "total_max": round(max(total_times), 3),
        },
        "data": _timing_log,
    }


@app.get("/api/timing/csv")
async def get_timing_csv():
    """タイミングログをCSV形式で返す。"""
    from fastapi.responses import PlainTextResponse
    if not _timing_log:
        return PlainTextResponse("text,slide,asr_sec,llm_sec,total_sec,confidence,element_id\n")
    lines = ["text,slide,asr_sec,llm_sec,total_sec,confidence,element_id"]
    for r in _timing_log:
        lines.append(
            f'"{r["text"]}",{r["slide"]},{r["asr_sec"]},'
            f'{r["llm_sec"]},{r["total_sec"]},{r["confidence"]},{r["element_id"]}'
        )
    return PlainTextResponse("\n".join(lines), media_type="text/csv")


@app.get("/api/pptx/status")
async def get_pptx_status():
    """PPTXアップロード・セットアップの進捗を返す。"""
    return _pptx_setup_status


@app.post("/api/upload/pptx")
async def upload_pptx(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    PPTXファイルをアップロードし、バックグラウンドでセットアップを開始する。
    進捗は Socket.IO の pptx_progress イベントで通知される。
    """
    global _pptx_setup_status

    if _pptx_setup_status["state"] == "processing":
        return JSONResponse(
            status_code=409,
            content={"error": "セットアップ中です。完了までお待ちください。"}
        )

    if not file.filename.lower().endswith(".pptx"):
        return JSONResponse(
            status_code=400,
            content={"error": ".pptx ファイルのみ対応しています"}
        )

    # アップロードファイルを一時保存
    upload_dir = Path(__file__).parent / "server_output" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / file.filename

    content_bytes = await file.read()
    with open(save_path, "wb") as f:
        f.write(content_bytes)

    logger.info(f"[UPLOAD] PPTXアップロード受信: {file.filename} ({len(content_bytes)//1024}KB)")

    # バックグラウンドでセットアップ実行
    background_tasks.add_task(_run_pptx_setup, str(save_path))

    return {"status": "accepted", "filename": file.filename}


async def _run_pptx_setup(pptx_path: str):
    """PPTXセットアップをバックグラウンドで実行し、進捗をSocket.IOで通知する。"""
    global _pptx_setup_status, _slide_data, _images_dir, _asr_prompt_cache, _asr_prompt_ready

    async def _notify(state: str, message: str, progress: int):
        _pptx_setup_status["state"]   = state
        _pptx_setup_status["message"] = message
        _pptx_setup_status["progress"] = progress
        await sio.emit("pptx_progress", _pptx_setup_status)
        logger.info(f"[PPTX_SETUP] {state} ({progress}%): {message}")

    try:
        await _notify("processing", "PPTXを解析中...", 10)

        from mvp_test import setup_from_pptx

        # ブロッキング処理をスレッドで実行
        loop = asyncio.get_event_loop()
        output_dir = str(Path(pptx_path).parent.parent)

        def _setup_with_coinit():
            # COMオートメーション（PowerPoint）はスレッドごとにCoInitializeが必要
            import comtypes
            comtypes.CoInitialize()
            try:
                return setup_from_pptx(pptx_path, output_dir=output_dir)
            finally:
                comtypes.CoUninitialize()

        slide_json, image_paths = await loop.run_in_executor(
            None,
            _setup_with_coinit
        )

        await _notify("processing", "スライドデータを読み込み中...", 80)

        # グローバル状態を更新
        _slide_data  = load_slide_data(slide_json)
        _images_dir  = str(Path(image_paths[0]).parent) if image_paths else ""
        _asr_prompt_cache = {}
        _asr_prompt_ready = set()

        slide_count = len(_slide_data.get("slides", []))
        await _notify("processing", f"ASRプロンプトを生成中... ({slide_count}スライド)", 90)

        # ASR prompt生成をバックグラウンドで開始
        t = threading.Thread(
            target=_generate_all_asr_prompts, daemon=True, name="asr-prompt-gen"
        )
        t.start()

        await _notify("done", f"セットアップ完了 ({slide_count}スライド)", 100)

        # スライド1に初期化してフロントエンドに通知
        await sio.emit("slide_changed", {"slide_num": 1})
        await sio.emit("slides_loaded", {
            "total": slide_count,
            "slide_width_pt":  _slide_data.get("slide_width_pt"),
            "slide_height_pt": _slide_data.get("slide_height_pt"),
        })

        logger.info(f"[PPTX_SETUP] 完了: {slide_count}スライド")

    except Exception as e:
        logger.error(f"[PPTX_SETUP] エラー: {e}", exc_info=True)
        await _notify("error", f"エラー: {type(e).__name__}: {e}", 0)


@app.get("/api/slide/{slide_num}/elements")
async def get_slide_elements(slide_num: int):
    """スライドの要素メタデータを返す。"""
    elements = get_elements_for_slide(_slide_data, slide_num)
    if elements is None:
        raise HTTPException(status_code=404, detail=f"スライド {slide_num} が見つかりません")
    # bbox_ratio のみフロントエンドに渡す（VLM説明文など不要なデータを省く）
    slim = []
    for e in elements:
        slim.append({
            "id": e["id"],
            "type": e["type"],
            "content": e.get("content"),
            "bbox_ratio": e.get("bbox_ratio"),
            "shape_name": e.get("shape_name"),
        })
    return {"slide_num": slide_num, "elements": slim}


# ============================================================
# Socket.IOイベントハンドラ
# ============================================================

@sio.event
async def connect(sid, environ):
    logger.info(f"[WS] クライアント接続: {sid}")
    # スライドがロード済みの場合のみ現在のスライド番号を送信
    if _slide_data:
        await sio.emit("slide_changed", {"slide_num": _current_slide}, room=sid)
    else:
        # 未ロード時はアップロード待ち状態を送信
        await sio.emit("pptx_progress", {
            "state": "idle",
            "message": "PPTXファイルを選択してください",
            "progress": 0,
        }, room=sid)


@sio.event
async def disconnect(sid):
    logger.info(f"[WS] クライアント切断: {sid}")


@sio.event
async def audio_chunk(sid, data):
    """
    ブラウザからの音声チャンクを受信してWhisperで文字起こしする。
    data: { audio: <base64 encoded PCM float32 mono 16kHz>, slide_num: int }
    サーバー側マイク（--asr）不要で動作する。
    """
    global _browser_asr_model

    if not _slide_data or "slides" not in _slide_data:
        return

    audio_b64  = data.get("audio")
    slide_num  = data.get("slide_num", _current_slide)

    if not audio_b64:
        return

    # Whisperモデルの遅延ロード（初回のみ）
    if _browser_asr_model is None:
        with _browser_asr_lock:
            if _browser_asr_model is None:
                try:
                    from faster_whisper import WhisperModel
                    logger.info(f"[BROWSER_ASR] Whisperモデルロード中: {_asr_model_size}")
                    _browser_asr_model = WhisperModel(
                        _asr_model_size, device="cuda", compute_type="float16"
                    )
                    logger.info("[BROWSER_ASR] モデルロード完了")
                except Exception as e:
                    logger.error(f"[BROWSER_ASR] モデルロード失敗: {e}", exc_info=True)
                    await sio.emit("error", {"message": f"ASRモデルロード失敗: {e}"}, room=sid)
                    return

    # base64 → numpy配列に変換
    try:
        import base64, numpy as np
        audio_bytes = base64.b64decode(audio_b64)
        audio_np    = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        logger.error(f"[BROWSER_ASR] 音声デコード失敗: {e}")
        return

    # Whisperで文字起こし（run_in_executorでブロッキング回避）
    loop = asyncio.get_event_loop()
    try:
        def _transcribe():
            max_amp = float(abs(audio_np).max())

            # 無音チャンクはスキップ（幻聴防止）
            if max_amp < 0.01:
                return []

            segments, info = _browser_asr_model.transcribe(
                audio_np,
                beam_size=5,
                language="ja",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=_asr_initial_prompt,
            )
            results = []
            for seg in segments:
                text = seg.text.strip()
                logprob = seg.avg_logprob
                no_speech = getattr(seg, "no_speech_prob", 0)
                if logprob > -1.0 and no_speech < 0.5:
                    is_kanji = any("一" <= c <= "龯" for c in text)
                    if len(text) > 1 or (len(text) == 1 and is_kanji):
                        results.append((text, logprob))
            return results

        t0 = time.time()
        results = await loop.run_in_executor(None, _transcribe)
        asr_elapsed = time.time() - t0

        for text, logprob in results:
            logger.info(f"[BROWSER_ASR] {text}  (logprob={logprob:.2f}, asr={asr_elapsed:.2f}s)")
            await sio.emit("asr_transcript", {"text": text, "confidence": round(logprob, 3)}, room=sid)
            _trigger_pointing(text, asr_elapsed=asr_elapsed)

    except Exception as e:
        logger.error(f"[BROWSER_ASR] 文字起こし失敗: {e}", exc_info=True)


@sio.event
async def change_slide(sid, data):
    """スライド切り替えイベント。"""
    global _current_slide, _asr_initial_prompt
    slide_num = data.get("slide_num", 1)
    _current_slide = slide_num
    # ASR initial_prompt をスライドに合わせて更新
    _asr_initial_prompt = _build_asr_initial_prompt(slide_num)
    logger.info(f"[WS] スライド切り替え: {slide_num}  prompt={_asr_initial_prompt[:40]}...")
    await sio.emit("slide_changed", {"slide_num": slide_num})


@sio.event
async def asr_control(sid, data):
    """
    マイクASRのON/OFF制御イベント。
    data: { action: "start" | "stop" }
    """
    action = data.get("action", "")
    if action == "start":
        if not _asr_enabled:
            await sio.emit("error", {"message": "ASRは --asr フラグなしでは使用できません"}, room=sid)
            return
        _start_asr_thread()
        await sio.emit("asr_status", {"running": _asr_running})
    elif action == "stop":
        _stop_asr_thread()
        await sio.emit("asr_status", {"running": False})


@sio.event
async def asr_text(sid, data):
    """
    ASRテキストを受信し、semantic_pointer で意味理解を実行してポインティング結果を返す。
    data: { text: str, slide_num: int }
    """
    global _processing

    text = data.get("text", "").strip()
    slide_num = data.get("slide_num", _current_slide)

    if not text:
        return

    logger.info(f"[WS] ASRテキスト受信 (slide={slide_num}): {text}")

    # 二重実行防止
    if _processing:
        await sio.emit("status", {"message": "⏳ 前のリクエストを処理中..."}, room=sid)
        return

    _processing = True
    await sio.emit("status", {"message": "🤔 意味解析中..."}, room=sid)

    try:
        # 非同期でLLM推論を実行（ブロッキングを避けるためrun_in_executor使用）
        loop = asyncio.get_event_loop()
        elements = get_elements_for_slide(_slide_data, slide_num)

        if not elements:
            await sio.emit("error", {"message": f"スライド {slide_num} の要素が見つかりません"}, room=sid)
            return

        result = await loop.run_in_executor(
            None,
            lambda: get_pointing_target(text, elements, model=_llm_model)
        )

        logger.info(f"[WS] ポインティング結果: id={result.get('element_id')}, "
              f"conf={result.get('confidence')}")

        # フロントエンドに送信
        await sio.emit("pointing_result", {
            "asr_text": text,
            "slide_num": slide_num,
            **result
        })

        if result.get("confidence") == "none":
            await sio.emit("status", {"message": "💭 ポインティング対象なし"})
        else:
            conf_emoji = {"high": "🎯", "medium": "📍", "low": "📌"}.get(
                result.get("confidence"), "📌")
            preview = result.get("content_preview", "")[:20]
            await sio.emit("status", {
                "message": f"{conf_emoji} 「{preview}」にポインティング"
            })

    except Exception as e:
        logger.error(f"[WS] 予期しないエラー: {e}", exc_info=True)
        await sio.emit("error", {"message": f"サーバーエラー: {type(e).__name__}: {e}"}, room=sid)

    finally:
        _processing = False


# ============================================================
# ユーティリティ
# ============================================================

# ============================================================
# ASRスレッド（タスク2.3）
# ============================================================

def _emit_from_thread(event: str, data: dict):
    """
    バックグラウンドスレッドからSocket.IOイベントを安全にemitする。
    asyncio.run_coroutine_threadsafeを使ってメインループにコルーチンをスケジュールする。
    """
    if _main_loop is None or _main_loop.is_closed():
        return
    future = asyncio.run_coroutine_threadsafe(
        sio.emit(event, data),
        _main_loop
    )
    try:
        future.result(timeout=2.0)
    except Exception:
        pass  # タイムアウト時は無視（次のチャンクで再試行される）


def _asr_worker():
    """
    ASRワーカースレッド本体。
    asr_demo.py のロジックを server.py 用に移植したもの。

    処理フロー:
        1. faster-whisper モデルをロード（初回のみ）
        2. PyAudio でマイク入力を開始
        3. RECORD_SECONDS ごとに音声チャンクをWhisperに渡す
        4. 認識結果を asr_transcript イベントで emit
        5. 信頼度が閾値を超えた発話は _handle_asr_result() でポインティングを起動
        6. _asr_stop_event がセットされたら終了
    """
    global _asr_running

    # --- 設定 ---
    CHANNELS       = 1
    RATE           = 16000
    CHUNK          = 1024
    RECORD_SECONDS = 3      # 3秒ごとに文字起こし

    # --- フィルタ設定 ---
    LOG_PROB_THRESHOLD   = -1.0   # avg_logprobの閾値（これ以下は破棄）
    NO_SPEECH_THRESHOLD  = 0.5    # no_speech_probの閾値（0.6→0.5に厳しく）

    # 幻聴・フィラーブラックリスト
    import re as _re
    _HALLUCINATION_PATTERNS = _re.compile(
        # YouTube系幻聴
        r"ご視聴|チャンネル登録|高評価|字幕|ありがとうございました|お願いいたします"
        r"|subscribe|please like|thank you for watching"
        r"|ご清聴|拍手|BGM|♪|…+"
        # 相槌・フィラー（単独で出た場合のみ弾く）
        r"|^(はい|いいえ|うん|えー+|あー+|えっと|そうですね|なるほど|わかりました)[。、]*$",
        _re.IGNORECASE,
    )

    try:
        import pyaudio
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as e:
        _emit_from_thread("error", {"message": f"ASR依存ライブラリ未インストール: {e}"})
        _asr_running = False
        return

    # スライドコンテキストを initial_prompt に使用（精度向上）
    # グローバル変数 _asr_initial_prompt をワーカーが参照し、
    # スライド切り替え時にメインループ側から更新する
    global _asr_initial_prompt
    _asr_initial_prompt = _build_asr_initial_prompt()

    logger.info(f"[ASR] モデルロード中: {_asr_model_size} / cuda / float16")
    _emit_from_thread("status", {"message": f"🔄 Whisper {_asr_model_size} をロード中..."})

    try:
        model = WhisperModel(_asr_model_size, device="cuda", compute_type="float16")
    except Exception:
        # CUDAが使えない場合はCPUにフォールバック
        logger.info("[ASR] CUDA不可 → CPUにフォールバック")
        model = WhisperModel(_asr_model_size, device="cpu", compute_type="int8")

    logger.info("[ASR] モデルロード完了。マイク入力開始...")
    _emit_from_thread("asr_status", {"running": True})
    _emit_from_thread("status", {"message": "🎤 マイク入力中..."})

    audio_if = pyaudio.PyAudio()
    stream   = audio_if.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = collections.deque()
    last_t = time.time()
    _asr_running = True

    try:
        while not _asr_stop_event.is_set():
            data_chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data_chunk)

            if time.time() - last_t < RECORD_SECONDS:
                continue

            # --- 文字起こし ---
            audio_bytes = b"".join(list(frames))
            audio_np    = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            frames.clear()
            last_t = time.time()

            t_asr_start = time.time()
            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                language="ja",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=_asr_initial_prompt,  # スライド切り替えで動的更新される
            )
            asr_elapsed = time.time() - t_asr_start

            for seg in segments:
                text    = seg.text.strip()
                logprob = seg.avg_logprob
                no_speech_prob = getattr(seg, "no_speech_prob", 0.0)

                # フィルタA: avg_logprob（信頼度）
                if logprob <= LOG_PROB_THRESHOLD:
                    continue

                # フィルタB: no_speech_prob（Whisperが「無音」と判定した確率）
                if no_speech_prob >= NO_SPEECH_THRESHOLD:
                    logger.info(f"[ASR] 無音判定スキップ: {text[:30]}  (no_speech={no_speech_prob:.2f})")
                    continue

                # フィルタC: 幻聴ブラックリスト（YouTube系定型句）
                if _HALLUCINATION_PATTERNS.search(text):
                    logger.info(f"[ASR] 幻聴スキップ: {text[:40]}")
                    continue

                # 短すぎるテキストフィルタ
                is_kanji = any('\u4e00' <= c <= '\u9faf' for c in text)
                if len(text) <= 1 and not (len(text) == 1 and is_kanji):
                    continue

                logger.info(f"[ASR] {text}  (logprob={logprob:.2f}, no_speech={no_speech_prob:.2f}, asr={asr_elapsed:.2f}s)")

                # 字幕表示用にフロントへ送信
                _emit_from_thread("asr_transcript", {
                    "text":       text,
                    "confidence": round(logprob, 3),
                })

                # 意味理解パイプラインを非同期で起動（ASR時間を渡す）
                _trigger_pointing(text, asr_elapsed=asr_elapsed)

    except Exception as e:
        logger.error(f"[ASR] 予期しないエラー: {e}", exc_info=True)
        _emit_from_thread("error", {"message": f"ASRエラー: {type(e).__name__}: {e}"})
    finally:
        stream.stop_stream()
        stream.close()
        audio_if.terminate()
        _asr_running = False
        _emit_from_thread("asr_status", {"running": False})
        _emit_from_thread("status", {"message": "⏹ マイク入力停止"})
        logger.info("[ASR] ワーカースレッド終了")


def _build_asr_initial_prompt(slide_num: int | None = None) -> str:
    """
    指定スライドの asr_prompt を返す。

    優先順位:
      1. _asr_prompt_cache（バックグラウンド生成済みのLLMキャッシュ）
      2. slide_data の asr_prompt フィールド（slide_parser --gen-asr-prompts で生成済みの場合）
      3. フォールバック: スライドのテキスト要素を単純結合
    """
    target = slide_num if slide_num is not None else _current_slide
    if not _slide_data:
        return "プレゼンテーション スライド 発表"

    # 優先1: バックグラウンド生成キャッシュ
    if target in _asr_prompt_cache:
        return _asr_prompt_cache[target]

    # 優先2: JSONに保存済みのasr_prompt
    for s in _slide_data.get("slides", []):
        if s["slide_number"] == target:
            if "asr_prompt" in s and s["asr_prompt"]:
                return s["asr_prompt"]
            break

    # フォールバック: テキストを単純結合（従来の動作）
    elements = get_elements_for_slide(_slide_data, target)
    texts = []
    for e in (elements or []):
        content = e.get("content")
        if isinstance(content, list):
            texts.extend(content)
        elif isinstance(content, str) and content.strip():
            texts.append(content.strip())

    prompt = "プレゼンテーション スライド 発表 " + " ".join(texts)
    return prompt[:224]  # Whisperのpromptは224トークン上限


def _trigger_pointing(text: str, asr_elapsed: float = 0.0):
    """
    ASRスレッドから呼ばれ、ポインティング処理を asyncio ループにスケジュールする。

    スロットリングロジック（タスク2.4）:
      1. 同一テキストが _COOLDOWN_SEC 以内に再送された場合は無視
      2. LLM処理中（_processing=True）の場合は _pending_text に上書き保存し、
         完了後に処理する（完全スキップではなく「最新の発話」を1件保持）
    """
    global _last_asr_text, _last_asr_time, _pending_text

    now = time.time()

    # クールダウン: 同一テキストの連続送信を無視
    if text == _last_asr_text and (now - _last_asr_time) < _COOLDOWN_SEC:
        logger.debug(f"[THROTTLE] クールダウン中のためスキップ: {text[:30]}")
        return

    _last_asr_text = text
    _last_asr_time = now

    if _main_loop is None or _main_loop.is_closed():
        return

    # スライドデータ未ロード時は無視
    if not _slide_data or "slides" not in _slide_data:
        return

    if _processing:
        # LLM処理中: 最新発話をキューに保持（上書き）
        _pending_text = text
        logger.debug(f"[THROTTLE] LLM処理中のためキューに保持: {text[:30]}")
        return

    asyncio.run_coroutine_threadsafe(
        _run_pointing(text, _current_slide, asr_elapsed=asr_elapsed),
        _main_loop
    )


async def _run_pointing(text: str, slide_num: int, asr_elapsed: float = 0.0):
    """
    意味理解 → Socket.IO emit の非同期コルーチン。
    asr_text イベントハンドラと同じロジックを共有する。
    """
    global _processing, _pending_text, _timing_log

    if _processing:
        _pending_text = text
        return

    # スライドデータ未ロード時は無視
    if not _slide_data or "slides" not in _slide_data:
        logger.debug("[POINT] スライド未ロードのためスキップ")
        return

    _processing = True
    await sio.emit("status", {"message": "🤔 意味解析中..."})

    t_llm_start = time.time()

    try:
        loop     = asyncio.get_event_loop()
        elements = get_elements_for_slide(_slide_data, slide_num)

        if not elements:
            return

        result = await loop.run_in_executor(
            None,
            lambda: get_pointing_target(text, elements, model=_llm_model)
        )

        llm_elapsed   = time.time() - t_llm_start
        total_elapsed = asr_elapsed + llm_elapsed

        logger.info(f"[POINT] id={result.get('element_id')}, conf={result.get('confidence')}")
        logger.info(f"[TIMING] ASR={asr_elapsed:.2f}s | LLM={llm_elapsed:.2f}s | Total={total_elapsed:.2f}s | slide={slide_num}")

        # タイミングログを記録（最大100件）
        _timing_log.append({
            "text":          text[:40],
            "slide":         slide_num,
            "asr_sec":       round(asr_elapsed, 3),
            "llm_sec":       round(llm_elapsed, 3),
            "total_sec":     round(total_elapsed, 3),
            "confidence":    result.get("confidence"),
            "element_id":    result.get("element_id"),
        })
        if len(_timing_log) > 100:
            _timing_log.pop(0)

        await sio.emit("pointing_result", {
            "asr_text":  text,
            "slide_num": slide_num,
            **result,
        })

        # image要素の場合、chart_pointingを非同期で実行
        # vlm_descriptionなし（写真）の場合はchart_pointingをスキップ
        if result.get("is_image_element"):
            if result.get("has_vlm_description"):
                asyncio.ensure_future(
                    _run_chart_pointing(text, result, slide_num)
                )
            # vlm_descriptionなしの写真 → chart_pointingなし、bbox全体でポインティングのみ

        # タイミング情報もフロントに送信
        await sio.emit("timing", {
            "asr_sec":   round(asr_elapsed, 3),
            "llm_sec":   round(llm_elapsed, 3),
            "total_sec": round(total_elapsed, 3),
        })

        if result.get("confidence") == "none":
            await sio.emit("status", {"message": "💭 ポインティング対象なし"})
        else:
            emoji   = {"high": "🎯", "medium": "📍", "low": "📌"}.get(
                result.get("confidence"), "📌")
            preview = result.get("content_preview", "")[:20]
            await sio.emit("status", {"message": f"{emoji} 「{preview}」にポインティング"})

    except Exception as e:
        logger.error(f"[POINT] 予期しないエラー: {e}", exc_info=True)
        await sio.emit("error", {"message": f"ポインティングエラー: {type(e).__name__}: {e}"})
    finally:
        _processing = False
        # キューに保留中の発話があれば処理
        if _pending_text is not None:
            pending = _pending_text
            _pending_text = None
            logger.debug(f"[THROTTLE] キュー処理: {pending[:30]}")
            asyncio.ensure_future(_run_pointing(pending, _current_slide))

async def _run_chart_pointing(asr_text: str, pointing_result: dict, slide_num: int):
    """
    chart_pointingをバックグラウンドで実行し、完了後にフロントへ送信する。
    llama3のpointing_resultとは独立して非同期実行される。
    """
    import base64

    element_id = pointing_result.get("element_id")
    if not element_id or not _slide_data:
        return

    # スライドデータから対象要素を取得
    elements = get_elements_for_slide(_slide_data, slide_num)
    matched  = next((e for e in elements if e["id"] == element_id), None)
    if not matched or matched["type"] != "image":
        return

    vlm_desc   = matched.get("vlm_description")
    bbox_ratio = matched.get("bbox_ratio")
    if not vlm_desc or not bbox_ratio:
        return

    # スライド画像を読み込んでbase64化
    image_b64 = None
    image_path = _get_image_path(slide_num)
    if image_path and os.path.exists(image_path):
        try:
            from PIL import Image
            import io
            img = Image.open(image_path).convert("RGB")
            # 対象要素のbboxでクロップ
            W, H = img.size
            rx, ry, rw, rh = bbox_ratio
            crop = img.crop((
                int(rx * W), int(ry * H),
                int((rx + rw) * W), int((ry + rh) * H)
            ))
            # 最大辺1024pxにリサイズ
            MAX_SIZE = 1024
            if max(crop.size) > MAX_SIZE:
                ratio    = MAX_SIZE / max(crop.size)
                new_size = (int(crop.size[0] * ratio), int(crop.size[1] * ratio))
                crop     = crop.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"[CHART_POINTING] 画像読み込み失敗: {e}")

    logger.info(f"[CHART_POINTING] 開始: {element_id} / image={'あり' if image_b64 else 'なし'}")
    t0 = time.time()

    loop = asyncio.get_event_loop()
    try:
        chart_result = await loop.run_in_executor(
            None,
            lambda: get_chart_pointing(
                asr_text=asr_text,
                vlm_description=vlm_desc,
                image_bbox_ratio=bbox_ratio,
                image_b64=image_b64,
                model="ministral-3:8b",
                verbose=True,
            )
        )
    except Exception as e:
        logger.error(f"[CHART_POINTING] エラー: {e}")
        return

    elapsed = time.time() - t0
    logger.info(f"[CHART_POINTING] 完了: {elapsed:.2f}s / draw_type={chart_result.get('draw_type') if chart_result else None}")

    if chart_result and chart_result.get("draw_type") != "none":
        await sio.emit("chart_pointing_result", {
            "asr_text":   asr_text,
            "slide_num":  slide_num,
            "element_id": element_id,
            **chart_result,
        })

def _generate_all_asr_prompts():
    """
    バックグラウンドスレッドで全スライドのASR initial_promptをLLMで生成し
    _asr_prompt_cache に格納する。

    - サーバー起動直後にlifespanから呼ばれる（非ブロッキング）
    - 生成完了したスライドから随時 _asr_initial_prompt を更新
    - 現在表示中のスライドを最優先で処理する
    - ポインティング用LLM（Ollama）と同じインスタンスを使うため、
      処理中はポインティングの応答が若干遅れる可能性がある
    """
    global _asr_prompt_cache, _asr_prompt_ready, _asr_initial_prompt

    try:
        import ollama as _ollama
    except ImportError:
        logger.info("[ASR_PROMPT] ollama が利用できないためprompt生成をスキップします。")
        return

    slides = _slide_data.get("slides", [])
    if not slides:
        return

    total = len(slides)
    logger.info(f"[ASR_PROMPT] バックグラウンドでprompt生成開始 ({total}スライド / モデル: {_asr_prompt_model})")

    SYSTEM = (
        "You are a helper for a Japanese speech recognition system.\n"
        "Given slide content, output a SHORT Japanese passage (2-3 sentences, max 180 chars)\n"
        "that naturally contains ALL technical terms, proper nouns, and key numbers from the slide.\n"
        "This will be used as Whisper's initial_prompt to improve recognition of domain-specific words.\n"
        "Rules: natural Japanese sentences (NOT a list), include every keyword, under 180 chars.\n"
        "Respond ONLY with the Japanese passage."
    )

    # 現在のスライドを先頭に並べ替えて優先処理
    ordered = sorted(slides, key=lambda s: (s["slide_number"] != _current_slide, s["slide_number"]))

    for slide in ordered:
        slide_num = slide["slide_number"]

        # テキスト収集
        texts = []
        for e in slide.get("elements", []):
            content = e.get("content")
            is_title = ("タイトル" in e.get("shape_name", "") or
                        "Title" in e.get("shape_name", ""))
            if isinstance(content, list):
                entry = " ".join(content)
            elif isinstance(content, str) and content.strip():
                entry = content.strip()
            else:
                entry = e.get("vlm_description", "") or ""
            if not entry:
                continue
            texts.insert(0, entry) if is_title else texts.append(entry)

        if not texts:
            _asr_prompt_cache[slide_num] = "プレゼンテーション スライド 発表"
            _asr_prompt_ready.add(slide_num)
            continue

        slide_text = "\n".join(texts)
        if len(slide_text) > 600:
            slide_text = slide_text[:600] + "..."

        user_msg = (
            f"スライド{slide_num}の内容：\n{slide_text}\n\n"
            "このスライドの専門用語・固有名詞・数値を全て含む自然な日本語文（2〜3文、180文字以内）を生成してください。"
        )

        try:
            resp = _ollama.chat(
                model=_llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                options={"num_predict": 120, "temperature": 0.3},
            )
            prompt = resp["message"]["content"].strip()
            if len(prompt) > 200:
                prompt = prompt[:200]

            _asr_prompt_cache[slide_num] = prompt
            _asr_prompt_ready.add(slide_num)

            # 現在表示中のスライドのpromptが完成したらすぐに反映
            if slide_num == _current_slide:
                _asr_initial_prompt = prompt
                logger.info(f"[ASR_PROMPT] ★現在スライド({slide_num})のprompt完成: {prompt[:50]}...")
            else:
                logger.info(f"[ASR_PROMPT]  スライド{slide_num:2d} 完了: {prompt[:50]}...")

        except Exception as e:
            logger.info(f"[ASR_PROMPT]  スライド{slide_num:2d} エラー: {e}")
            # エラー時はフォールバックをキャッシュ
            fallback = "プレゼンテーション スライド 発表 " + " ".join(texts[:3])
            _asr_prompt_cache[slide_num] = fallback[:200]
            _asr_prompt_ready.add(slide_num)

        # スライドごとの進捗をフロントエンドに通知
        _emit_from_thread("asr_prompt_progress", {
            "done":  len(_asr_prompt_ready),
            "total": total,
            "slide": slide_num,
        })

    logger.info(f"[ASR_PROMPT] 全スライドのprompt生成完了 ({len(_asr_prompt_ready)}/{total})")
    # 全完了をフロントエンドに通知
    _emit_from_thread("asr_prompt_ready", {
        "done":  len(_asr_prompt_ready),
        "total": total,
    })


def _start_asr_thread():
    """ASRワーカースレッドを起動する。既に起動中の場合は何もしない。"""
    global _asr_thread, _asr_running

    if _asr_running:
        logger.info("[ASR] すでに動作中です")
        return

    _asr_stop_event.clear()
    _asr_thread = threading.Thread(target=_asr_worker, daemon=True, name="asr-worker")
    _asr_thread.start()
    logger.info("[ASR] ワーカースレッド起動")


def _stop_asr_thread():
    """ASRワーカースレッドを停止する。"""
    global _asr_running

    if not _asr_running:
        return

    logger.info("[ASR] 停止シグナル送信...")
    _asr_stop_event.set()

    if _asr_thread and _asr_thread.is_alive():
        _asr_thread.join(timeout=5.0)

    _asr_running = False
    logger.info("[ASR] ワーカースレッド停止完了")


# ============================================================
# ユーティリティ
# ============================================================

def _get_image_path(slide_num: int) -> str | None:
    """スライド番号から画像パスを返す。"""
    # slide_dataにimage_pathが埋め込まれている場合
    for s in _slide_data.get("slides", []):
        if s["slide_number"] == slide_num:
            if "image_path" in s and os.path.exists(s["image_path"]):
                return s["image_path"]
            break

    # images_dirから推測
    if _images_dir:
        candidates = [
            os.path.join(_images_dir, f"slide_{slide_num:03d}.png"),
            os.path.join(_images_dir, f"slide{slide_num}.png"),
            os.path.join(_images_dir, f"{slide_num:03d}.png"),
            os.path.join(_images_dir, f"Slide{slide_num}.PNG"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    return None


# ============================================================
# 起動処理
# ============================================================

def load_resources(slide_json_path: str, images_dir: str, model: str,
                   asr_prompt_model: str = "llama3",
                   asr: bool = False, asr_model: str = "small"):
    """スライドデータと設定を読み込む。"""
    global _slide_data, _images_dir, _llm_model, _asr_prompt_model, _asr_enabled, _asr_model_size

    if not os.path.exists(slide_json_path):
        sys.exit(f"[ERROR] JSONが見つかりません: {slide_json_path}")

    logger.info(f"[INFO] スライドJSON読み込み: {slide_json_path}")
    _slide_data = load_slide_data(slide_json_path)

    slide_count = len(_slide_data.get("slides", []))
    logger.info(f"[INFO] スライド数: {slide_count}")

    _images_dir       = images_dir
    _llm_model        = model
    _asr_prompt_model = asr_prompt_model
    _asr_enabled      = asr
    _asr_model_size   = asr_model

    # --- 起動時の環境チェックログ ---
    logger.info("=" * 60)
    logger.info("Auto-Pointer サーバー 起動設定")
    logger.info("=" * 60)
    logger.info(f"  スライドJSON   : {slide_json_path}")
    logger.info(f"  スライド数     : {slide_count}")
    logger.info(f"  画像ディレクトリ: {images_dir or '(JSONのimage_pathを使用)'}")
    logger.info(f"  LLMモデル      : {model} (ポインティング推論)")
    logger.info(f"  ASR Promptモデル: {asr_prompt_model} (プロンプト生成)")
    logger.info(f"  マイクASR      : {'有効' if asr else '無効'}" + (f" (Whisper {asr_model})" if asr else ""))

    # GPU可用性チェック
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        logger.info(f"  GPU            : {'✓ ' + gpu_name if gpu_available else '✗ CPU fallback'}")
    except ImportError:
        logger.info("  GPU            : torch未インストール (Whisperはcudatoolkitを直接使用)")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Pointing プレゼンテーションサーバー"
    )
    parser.add_argument("--slide-json", default=None,
                        help="slide_parser.py の出力JSONパス")
    parser.add_argument("--images-dir", default="",
                        help="スライド画像ディレクトリ")
    parser.add_argument("--pptx", default=None,
                        help="PPTXを渡すと自動でJSON生成・画像エクスポートを行う")
    parser.add_argument("--model", default="llama3",
                        help="Ollamaモデル名 ポインティング推論用 (デフォルト: llama3)")
    parser.add_argument("--asr-prompt-model", default=None,
                        help="ASRプロンプト生成用モデル 精度重視 (デフォルト: --modelと同じ)")
    parser.add_argument("--asr", action="store_true",
                        help="マイクからのリアルタイムASRを有効化する")
    parser.add_argument("--asr-model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisperモデルサイズ (デフォルト: small)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="ホスト (デフォルト: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="ポート番号 (デフォルト: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="開発用ホットリロード（--pptx/--slide-jsonは固定）")
    args = parser.parse_args()

    # --pptx が指定された場合は自動セットアップ
    if args.pptx:
        try:
            from mvp_test import setup_from_pptx
        except ImportError:
            sys.exit("[ERROR] mvp_test.py が見つかりません。")

        if not os.path.exists(args.pptx):
            sys.exit(f"[ERROR] PPTXが見つかりません: {args.pptx}")

        logger.info(f"[INFO] PPTXを自動セットアップ中: {args.pptx}")
        slide_json, image_paths = setup_from_pptx(args.pptx, output_dir="server_output")
        images_dir = str(Path(image_paths[0]).parent) if image_paths else ""
        asr_pm = args.asr_prompt_model or args.model
        load_resources(slide_json, images_dir, args.model,
                       asr_prompt_model=asr_pm,
                       asr=args.asr, asr_model=args.asr_model)

    elif args.slide_json:
        asr_pm = args.asr_prompt_model or args.model
        load_resources(args.slide_json, args.images_dir, args.model,
                       asr_prompt_model=asr_pm,
                       asr=args.asr, asr_model=args.asr_model)
    else:
        # --pptx も --slide-json も指定なし → ブラウザからアップロードモードで起動
        logger.info("[INFO] PPTXファイル未指定 → ブラウザからアップロードしてください")
        _pptx_setup_status["state"]   = "idle"
        _pptx_setup_status["message"] = "PPTXファイルを選択してください"
        # ASR/LLM設定だけ先に適用（スライドなしでも起動直後からASR有効に）
        global _asr_enabled, _asr_model_size, _llm_model, _asr_prompt_model
        _asr_enabled      = args.asr
        _asr_model_size   = args.asr_model
        _llm_model        = args.model
        _asr_prompt_model = args.asr_prompt_model or args.model

    logger.info(f"\n🚀 サーバー起動: http://{args.host}:{args.port}")
    logger.info("   ブラウザで上記URLにアクセスしてください\n")

    uvicorn.run(
        sio_app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
