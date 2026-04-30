"""
server.py
=========
フェーズ2: WebSocketサーバー (FastAPI + python-socketio)

機能:
  - スライドJSONとスライド画像を読み込んでフロントエンドに配信
  - フロントエンドからASRテキストを受信し、semantic_pointer で意味理解を実行
  - ポインティング結果をリアルタイムでフロントエンドにブロードキャスト
  - スライド切り替えイベントの処理

起動方法:
    # 基本（カレントディレクトリのslide_data.jsonと画像を使用）
    python server.py --slide-json slide_data.json --images-dir slides_images

    # ポート指定
    python server.py --slide-json slide_data.json --images-dir slides_images --port 8000

    # PPTXから直接起動（slide_parser.pyが必要）
    python server.py --pptx your_slide.pptx

APIエンドポイント:
    GET  /                    → index.html (フロントエンド)
    GET  /api/slides          → スライドメタデータ一覧
    GET  /api/slide/{n}/image → スライド画像（PNG）
    GET  /api/slide/{n}/elements → スライド要素のJSON

Socket.IOイベント:
    クライアント → サーバー:
        asr_text     : { text: "発話テキスト", slide_num: 1 }
        change_slide : { slide_num: 1 }

    サーバー → クライアント:
        pointing_result : ポインティング結果 (semantic_pointer.py の返り値)
        slide_changed   : { slide_num: 1 }
        status          : { message: "処理中..." }
        error           : { message: "エラーメッセージ" }
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import socketio

# semantic_pointer を同じディレクトリから読み込む
sys.path.insert(0, str(Path(__file__).parent))
try:
    from semantic_pointer import (
        load_slide_data, get_elements_for_slide, get_pointing_target
    )
except ImportError:
    sys.exit("[ERROR] semantic_pointer.py が見つかりません。同じディレクトリに置いてください。")

# ============================================================
# グローバル状態
# ============================================================

_slide_data: dict = {}          # slide_parser.py の出力JSON
_images_dir: str  = ""          # スライド画像ディレクトリ
_current_slide: int = 1         # 現在のスライド番号（1-indexed）
_llm_model: str = "llama3"      # 使用するOllamaモデル
_processing: bool = False       # LLM処理中フラグ（二重実行防止）

# ============================================================
# FastAPI + Socket.IO セットアップ
# ============================================================

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

app = FastAPI(title="Auto-Pointing Presentation Server")

# Socket.IO を /ws にマウント
sio_app = socketio.ASGIApp(sio, socketio_path="socket.io")
app.mount("/ws", sio_app)

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
    print(f"[WS] クライアント接続: {sid}")
    # 接続時に現在のスライド番号を送信
    await sio.emit("slide_changed", {"slide_num": _current_slide}, room=sid)


@sio.event
async def disconnect(sid):
    print(f"[WS] クライアント切断: {sid}")


@sio.event
async def change_slide(sid, data):
    """スライド切り替えイベント。"""
    global _current_slide
    slide_num = data.get("slide_num", 1)
    _current_slide = slide_num
    print(f"[WS] スライド切り替え: {slide_num}")
    await sio.emit("slide_changed", {"slide_num": slide_num})


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

    print(f"[WS] ASRテキスト受信 (slide={slide_num}): {text}")

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

        print(f"[WS] ポインティング結果: id={result.get('element_id')}, "
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
        print(f"[WS] エラー: {e}")
        await sio.emit("error", {"message": str(e)}, room=sid)

    finally:
        _processing = False


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

def load_resources(slide_json_path: str, images_dir: str, model: str):
    """スライドデータと設定を読み込む。"""
    global _slide_data, _images_dir, _llm_model

    if not os.path.exists(slide_json_path):
        sys.exit(f"[ERROR] JSONが見つかりません: {slide_json_path}")

    print(f"[INFO] スライドJSON読み込み: {slide_json_path}")
    _slide_data = load_slide_data(slide_json_path)

    slide_count = len(_slide_data.get("slides", []))
    print(f"[INFO] スライド数: {slide_count}")

    _images_dir = images_dir
    _llm_model  = model
    print(f"[INFO] 画像ディレクトリ: {images_dir or '(slide_data.jsonのimage_pathを使用)'}")
    print(f"[INFO] LLMモデル: {model}")


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
                        help="Ollamaモデル名 (デフォルト: llama3)")
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

        print(f"[INFO] PPTXを自動セットアップ中: {args.pptx}")
        slide_json, image_paths = setup_from_pptx(args.pptx, output_dir="server_output")
        images_dir = str(Path(image_paths[0]).parent) if image_paths else ""
        load_resources(slide_json, images_dir, args.model)

    elif args.slide_json:
        load_resources(args.slide_json, args.images_dir, args.model)
    else:
        parser.error("--slide-json または --pptx を指定してください")

    print(f"\n🚀 サーバー起動: http://{args.host}:{args.port}")
    print("   ブラウザで上記URLにアクセスしてください\n")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
