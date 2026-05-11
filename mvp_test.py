"""
mvp_test.py
===========
フェーズ1 MVP統合テスト：
「録音音声 → ASR → 意味理解 → 座標特定 → ポインター描画」
の一連の流れをオフラインで検証するスクリプト。

使い方:
    # 音声ファイルを使ってフル検証
    python mvp_test.py \\
        --audio  test.wav \\
        --slide-json slide_data.json \\
        --slide-image slide5.png \\
        --slide-num 5

    # ASRをスキップしてテキスト直接入力（ASRなしで意味理解〜描画だけ確認）
    python mvp_test.py \\
        --text "意味的理解の表現について説明します" \\
        --slide-json slide_data.json \\
        --slide-image slide5.png \\
        --slide-num 5

    # 複数テキストをまとめて検証（テキストファイル）
    python mvp_test.py \\
        --text-file test_utterances.txt \\
        --slide-json slide_data.json \\
        --slide-image slide5.png \\
        --slide-num 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- 依存モジュール ---
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARNING] faster-whisper が見つかりません。--text モードのみ使用可能です。")

# semantic_pointer と draw_pointer を同じディレクトリから読み込む
sys.path.insert(0, str(Path(__file__).parent))
try:
    from semantic_pointer import (
        load_slide_data, get_elements_for_slide, get_pointing_target
    )
except ImportError:
    sys.exit("[ERROR] semantic_pointer.py が見つかりません。同じディレクトリに置いてください。")

try:
    from slide_parser import parse_pptx, export_slides_as_images
    SLIDE_PARSER_AVAILABLE = True
except ImportError:
    SLIDE_PARSER_AVAILABLE = False

try:
    from draw_pointer import apply_from_result
except ImportError:
    sys.exit("[ERROR] draw_pointer.py が見つかりません。同じディレクトリに置いてください。")


# ============================================================
# ASR（faster-whisper）
# ============================================================

def transcribe_audio(audio_path: str,
                     model_size: str = "small",
                     device: str = "cuda",
                     compute_type: str = "float16") -> list[str]:
    """
    音声ファイルをfaster-whisperで文字起こしし、テキストのリストを返す。
    各セグメントを1発話として扱う。
    """
    if not WHISPER_AVAILABLE:
        sys.exit("[ERROR] faster-whisper がインストールされていません。")

    print(f"[ASR] モデルロード中: {model_size} / {device} / {compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[ASR] 文字起こし開始: {audio_path}")
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="ja",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt="プレゼンテーション スライド 発表"
    )

    texts = []
    for seg in segments:
        text = seg.text.strip()
        # 信頼度フィルタ
        if seg.avg_logprob > -1.0:
            is_kanji = any('\u4e00' <= c <= '\u9faf' for c in text)
            if len(text) > 1 or (len(text) == 1 and is_kanji):
                texts.append(text)
                print(f"[ASR] [{seg.start:.1f}s→{seg.end:.1f}s] {text} "
                      f"(信頼度: {seg.avg_logprob:.2f})")

    elapsed = time.time() - t0
    print(f"[ASR] 完了: {len(texts)}セグメント / {elapsed:.1f}秒")
    return texts


# ============================================================
# パイプライン1発話分の処理
# ============================================================

def process_utterance(
    text: str,
    elements: list[dict],
    slide_image_path: str,
    output_dir: str,
    counter: int,
    style: str = "all",
    model: str = "llama3",
    verbose: bool = False,
) -> dict:
    """
    1発話分のASRテキストを受け取り、意味理解→描画までを実行する。
    結果のdictを返す。
    """
    print(f"\n{'='*60}")
    print(f"[{counter:03d}] ASR テキスト: {text}")
    print(f"{'='*60}")

    # --- 意味理解 ---
    t0 = time.time()
    result = get_pointing_target(text, elements, model=model, verbose=verbose)
    semantic_elapsed = time.time() - t0

    print(f"[SEMANTIC] element_id : {result.get('element_id')}")
    print(f"[SEMANTIC] confidence : {result.get('confidence')}")
    print(f"[SEMANTIC] reason     : {result.get('reason')}")
    print(f"[SEMANTIC] 処理時間   : {semantic_elapsed:.2f}秒")

    # --- ポインター描画 ---
    output_path = None
    draw_elapsed = 0.0
    if result.get("confidence") != "none":
        out_name = f"pointed_{counter:03d}.png"
        output_path = os.path.join(output_dir, out_name)

        t0 = time.time()
        apply_from_result(
            slide_image_path, result,
            style=style, show_label=True,
            output_path=output_path
        )
        draw_elapsed = time.time() - t0
        print(f"[DRAW]     出力画像  : {output_path}")
        print(f"[DRAW]     処理時間  : {draw_elapsed:.2f}秒")
    else:
        print("[DRAW]     ポインティングなし（confidence=none）")

    return {
        "index":            counter,
        "asr_text":         text,
        "element_id":       result.get("element_id"),
        "element_type":     result.get("element_type"),
        "content_preview":  result.get("content_preview"),
        "confidence":       result.get("confidence"),
        "reason":           result.get("reason"),
        "bbox_ratio":       result.get("bbox_ratio"),
        "output_image":     output_path,
        "semantic_elapsed": round(semantic_elapsed, 3),
        "draw_elapsed":     round(draw_elapsed, 3),
    }


# ============================================================
# メインパイプライン
# ============================================================

def run_pipeline(
    texts: list[str],
    slide_json_path: str,
    slide_image_path: str,
    slide_num: int,
    output_dir: str,
    style: str = "all",
    model: str = "llama3",
    verbose: bool = False,
) -> list[dict]:
    """全発話を順番に処理してレポートを返す。"""

    # スライドメタデータ読み込み
    slide_data = load_slide_data(slide_json_path)
    elements   = get_elements_for_slide(slide_data, slide_num)
    if not elements:
        sys.exit(f"[ERROR] スライド {slide_num} の要素が見つかりません")

    print(f"\n[INFO] スライド {slide_num} / 要素数: {len(elements)}")
    print(f"[INFO] 発話数: {len(texts)}")
    print(f"[INFO] モデル: {model} / スタイル: {style}")

    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_t0 = time.time()

    for i, text in enumerate(texts, start=1):
        r = process_utterance(
            text, elements, slide_image_path,
            output_dir, counter=i,
            style=style, model=model, verbose=verbose
        )
        results.append(r)

    total_elapsed = time.time() - total_t0

    # --- サマリー表示 ---
    print(f"\n{'='*60}")
    print("テスト結果サマリー")
    print(f"{'='*60}")
    pointed     = [r for r in results if r["confidence"] != "none"]
    not_pointed = [r for r in results if r["confidence"] == "none"]
    high_conf   = [r for r in results if r["confidence"] == "high"]
    avg_semantic = sum(r["semantic_elapsed"] for r in results) / len(results) if results else 0

    print(f"  総発話数             : {len(results)}")
    print(f"  ポインティングあり   : {len(pointed)} ({len(pointed)/len(results)*100:.0f}%)")
    print(f"  ポインティングなし   : {len(not_pointed)}")
    print(f"  高信頼度 (high)      : {len(high_conf)}")
    print(f"  平均意味理解時間     : {avg_semantic:.2f}秒/発話")
    print(f"  総処理時間           : {total_elapsed:.1f}秒")
    print(f"  出力ディレクトリ     : {output_dir}")

    print(f"\n{'─'*60}")
    print(f"{'#':>4}  {'信頼度':<8}  {'要素ID':<12}  発話テキスト")
    print(f"{'─'*60}")
    for r in results:
        conf = r['confidence'] or 'none'
        eid  = r['element_id'] or '-'
        print(f"  {r['index']:>3}  {conf:<8}  {eid:<12}  {r['asr_text'][:40]}")

    return results


# ============================================================
# PPTXからの自動セットアップ
# ============================================================

def setup_from_pptx(pptx_path: str,
                    output_dir: str,
                    use_vlm: bool = True,
                    image_width: int = 1920) -> tuple[str, list[str]]:
    """
    PPTXファイルからスライドJSONと画像を自動生成して返す。

    Returns
    -------
    tuple[str, list[str]]
        (slide_json_path, image_paths_list)
    """
    if not SLIDE_PARSER_AVAILABLE:
        sys.exit("[ERROR] slide_parser.py が見つかりません。同じディレクトリに置いてください。")

    import json as _json
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pptx_path).stem

    # 1. スライド解析 → JSON
    json_path = os.path.join(output_dir, f"{stem}_slide_data.json")
    print(f"[SETUP] スライド解析中: {pptx_path}")
    data = parse_pptx(pptx_path, use_vlm=use_vlm)
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SETUP] JSON保存: {json_path}")

    # 2. スライド画像エクスポート（COM）
    images_dir = os.path.join(output_dir, f"{stem}_images")
    print(f"[SETUP] スライド画像をエクスポート中...")
    image_paths = export_slides_as_images(pptx_path, images_dir, width_px=image_width)
    print(f"[SETUP] 画像エクスポート完了: {len(image_paths)}枚")

    # 3. JSONに画像パスを埋め込んで再保存
    for slide in data["slides"]:
        idx = slide["slide_index"]
        if idx < len(image_paths):
            slide["image_path"] = image_paths[idx]
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path, image_paths


# ============================================================
# CLIエントリポイント
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MVP統合テスト: ASR → 意味理解 → ポインター描画"
    )

    # 入力（どれか1つ必須）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio",     help="入力音声ファイル（.wav等）")
    input_group.add_argument("--text",      help="テスト発話テキスト（1文）")
    input_group.add_argument("--text-file", help="テスト発話テキストファイル（1行1発話）")

    # スライド指定：--pptx か (--slide-json + --slide-image) のどちらか
    parser.add_argument("--pptx", default=None,
                        help="PPTXを渡すと自動でJSON生成・画像エクスポートを行う")
    parser.add_argument("--slide-json",  default=None,
                        help="slide_parser.py の出力JSONパス（--pptx未使用時に必須）")
    parser.add_argument("--slide-image", default=None,
                        help="スライド画像のパス（--pptx未使用時に必須）")
    parser.add_argument("--slide-num",   type=int, default=1,
                        help="現在のスライド番号（1-indexed、デフォルト:1）")

    # オプション
    parser.add_argument("--output-dir", default="mvp_output",
                        help="出力ディレクトリ（デフォルト: mvp_output）")
    parser.add_argument("--style", default="all",
                        choices=["laser", "highlight", "border", "all"],
                        help="ポインタースタイル（デフォルト: all）")
    parser.add_argument("--model", default="llama3",
                        help="Ollamaモデル名（デフォルト: llama3）")
    parser.add_argument("--asr-model", default="small",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisperモデルサイズ（デフォルト: small）")
    parser.add_argument("--asr-device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="ASRデバイス（デフォルト: cuda）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="LLMのプロンプトと生レスポンスを表示")
    parser.add_argument("--save-report", action="store_true",
                        help="結果をJSONレポートとして保存する")

    args = parser.parse_args()

    # --pptx が指定された場合は自動セットアップ
    if args.pptx:
        if not os.path.exists(args.pptx):
            sys.exit(f"[ERROR] PPTXが見つかりません: {args.pptx}")
        slide_json_path, image_paths = setup_from_pptx(
            args.pptx, output_dir=args.output_dir)
        slide_idx = args.slide_num - 1
        if slide_idx >= len(image_paths):
            sys.exit(f"[ERROR] スライド {args.slide_num} の画像が見つかりません")
        slide_image_path = image_paths[slide_idx]
        print(f"[INFO] 使用スライド画像: {slide_image_path}")
    else:
        if not args.slide_json or not args.slide_image:
            sys.exit("[ERROR] --pptx を使わない場合は --slide-json と --slide-image が必要です")
        slide_json_path  = args.slide_json
        slide_image_path = args.slide_image

    # 入力テキストの準備
    if args.audio:
        texts = transcribe_audio(
            args.audio,
            model_size=args.asr_model,
            device=args.asr_device
        )
        if not texts:
            sys.exit("[ERROR] 音声から有効なテキストが取得できませんでした。")
    elif args.text:
        texts = [args.text]
    elif args.text_file:
        if not os.path.exists(args.text_file):
            sys.exit(f"[ERROR] テキストファイルが見つかりません: {args.text_file}")
        with open(args.text_file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    # パイプライン実行
    results = run_pipeline(
        texts=texts,
        slide_json_path=slide_json_path,
        slide_image_path=slide_image_path,
        slide_num=args.slide_num,
        output_dir=args.output_dir,
        style=args.style,
        model=args.model,
        verbose=args.verbose,
    )

    # レポート保存
    if args.save_report:
        report_path = os.path.join(args.output_dir, "mvp_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] レポート保存: {report_path}")


if __name__ == "__main__":
    main()
