"""
draw_pointer.py
===============
スライド画像の上に、semantic_pointer.py が返した bbox_ratio を元に
ポインター（レーザー円・ハイライト・矩形枠）を描画して画像として出力する。

使い方:
    # bbox_ratio を直接渡す（テスト用）
    python draw_pointer.py slide.png --bbox-ratio 0.078 0.1873 0.8545 0.7392 --style all

    # semantic_pointer.py の JSON出力ファイルを渡す
    python draw_pointer.py slide.png --result-json result.json --style laser

    # 対話モード（semantic_pointer と組み合わせた統合テスト用）
    python draw_pointer.py slide.png --interactive --slide-json slide_data.json --slide-num 5

スタイル:
    laser     : 赤い円（レーザーポインター風）
    highlight : 半透明の黄色塗りつぶし
    border    : 矩形枠線
    all       : 3つすべて重ねて描画
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("[ERROR] Pillow が見つかりません: pip install Pillow")


# ============================================================
# 描画設定
# ============================================================

LASER_COLOR      = (255, 40, 40, 220)    # 赤（RGBA）
LASER_RADIUS_RATIO = 0.025               # スライド幅に対するレーザー円の半径比率
LASER_GLOW_RATIO   = 0.045              # 外側グロー半径比率

HIGHLIGHT_COLOR  = (255, 230, 0, 90)     # 黄色半透明（RGBA）

BORDER_COLOR     = (255, 40, 40, 255)    # 赤（RGBA）
BORDER_WIDTH     = 4                     # ボーダー太さ（px）

LABEL_BG_COLOR   = (255, 40, 40, 200)
LABEL_TEXT_COLOR = (255, 255, 255, 255)


# ============================================================
# 描画関数
# ============================================================

def draw_laser(overlay: "Image.Image", cx: int, cy: int, slide_w: int) -> None:
    """レーザーポインター風の赤い円を描画する（グロー付き）。"""
    draw = ImageDraw.Draw(overlay, "RGBA")

    glow_r  = int(slide_w * LASER_GLOW_RATIO)
    laser_r = int(slide_w * LASER_RADIUS_RATIO)

    # 外側グロー（薄い赤）
    draw.ellipse(
        [cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
        fill=(255, 40, 40, 60)
    )
    # 中間グロー
    mid_r = (glow_r + laser_r) // 2
    draw.ellipse(
        [cx - mid_r, cy - mid_r, cx + mid_r, cy + mid_r],
        fill=(255, 40, 40, 120)
    )
    # コア（鮮明な赤）
    draw.ellipse(
        [cx - laser_r, cy - laser_r, cx + laser_r, cy + laser_r],
        fill=LASER_COLOR
    )


def draw_highlight(overlay: "Image.Image",
                   x: int, y: int, w: int, h: int) -> None:
    """半透明の黄色ハイライトをbbox全体に描画する。"""
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.rectangle([x, y, x + w, y + h], fill=HIGHLIGHT_COLOR)


def draw_border(overlay: "Image.Image",
                x: int, y: int, w: int, h: int,
                line_width: int = BORDER_WIDTH) -> None:
    """矩形枠線をbbox周囲に描画する。"""
    draw = ImageDraw.Draw(overlay, "RGBA")
    # 角を丸くするため複数の矩形で描画
    for i in range(line_width):
        draw.rectangle(
            [x + i, y + i, x + w - i, y + h - i],
            outline=BORDER_COLOR
        )


# 日本語フォントの候補パス（Windows / Mac / Linux）
_JP_FONT_CANDIDATES = [
    # Windows
    r"C:/Windows/Fonts/meiryo.ttc",
    r"C:/Windows/Fonts/msgothic.ttc",
    r"C:/Windows/Fonts/YuGothM.ttc",
    r"C:/Windows/Fonts/arial.ttf",
    # Mac
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def _get_font(font_size: int) -> "ImageFont.FreeTypeFont":
    """日本語対応フォントを優先して返す。見つからなければデフォルト。"""
    for path in _JP_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_label(overlay: "Image.Image",
               x: int, y: int, text: str, font_size: int = 18) -> None:
    """要素の上部にラベルテキストを描画する（デバッグ用）。"""
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _get_font(font_size)

    # テキストサイズ取得
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    padding = 4
    label_x = x
    label_y = max(0, y - th - padding * 2 - 2)

    # 背景矩形
    draw.rectangle(
        [label_x, label_y, label_x + tw + padding * 2, label_y + th + padding * 2],
        fill=LABEL_BG_COLOR
    )
    # テキスト
    draw.text(
        (label_x + padding, label_y + padding),
        text, fill=LABEL_TEXT_COLOR, font=font
    )


# ============================================================
# メイン描画関数
# ============================================================

def apply_pointer(
    slide_image_path: str,
    bbox_ratio: list[float],
    style: str = "all",
    label: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    スライド画像にポインターを描画して保存する。

    Parameters
    ----------
    slide_image_path : str
        入力スライド画像のパス（PNG/JPG）
    bbox_ratio : list[float]
        [x, y, w, h] の比率（0~1）。semantic_pointer.py の bbox_ratio と同形式。
    style : str
        "laser" / "highlight" / "border" / "all"
    label : str, optional
        要素の上部に表示するラベルテキスト
    output_path : str, optional
        出力先パス（省略時は入力ファイル名_pointed.png）

    Returns
    -------
    str
        出力ファイルのパス
    """
    img = Image.open(slide_image_path).convert("RGBA")
    slide_w, slide_h = img.size

    # bbox_ratio → ピクセル座標に変換
    rx, ry, rw, rh = bbox_ratio
    px = int(rx * slide_w)
    py = int(ry * slide_h)
    pw = int(rw * slide_w)
    ph = int(rh * slide_h)
    cx = px + pw // 2
    cy = py + ph // 2

    # 透明オーバーレイ
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    if style in ("highlight", "all"):
        draw_highlight(overlay, px, py, pw, ph)

    if style in ("border", "all"):
        draw_border(overlay, px, py, pw, ph)

    if style in ("laser", "all"):
        draw_laser(overlay, cx, cy, slide_w)

    if label:
        draw_label(overlay, px, py, label)

    # 合成
    result = Image.alpha_composite(img, overlay).convert("RGB")

    # 出力パス決定
    if not output_path:
        stem = Path(slide_image_path).stem
        output_path = str(Path(slide_image_path).parent / f"{stem}_pointed.png")

    result.save(output_path, "PNG")
    return output_path


# ============================================================
# semantic_pointer との統合ヘルパー
# ============================================================

def apply_from_result(
    slide_image_path: str,
    pointing_result: dict,
    style: str = "all",
    show_label: bool = True,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    semantic_pointer.py の返り値dictを受け取ってポインティング描画する。
    confidence が "none" の場合は何もせず None を返す。
    """
    if pointing_result.get("confidence") == "none":
        print("[DRAW] confidence=none のためポインティングをスキップします。")
        return None

    # SmartArtの場合はノード単位のbboxを優先、なければ全体bboxを使う
    node_bbox  = pointing_result.get("smartart_node_bbox_ratio")
    whole_bbox = pointing_result.get("bbox_ratio")

    if not whole_bbox:
        print("[DRAW] bbox_ratio が見つかりません。")
        return None

    # ラベルテキスト生成
    label = None
    if show_label:
        preview = pointing_result.get("content_preview", "")
        conf    = pointing_result.get("confidence", "")
        node_idx = pointing_result.get("smartart_node_index")
        prefix  = f"[node{node_idx}] " if node_idx is not None else ""
        short   = f"{preview[:28]}..." if len(preview) > 28 else preview
        label   = f"{prefix}{short} [{conf}]"

    overlay_img = Image.open(slide_image_path).convert("RGBA")
    slide_w, slide_h = overlay_img.size
    overlay = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))

    if node_bbox:
        # --- SmartArtノード単位の描画 ---
        # 全体に薄いボーダーを描画（コンテキスト表示）
        rx, ry, rw, rh = whole_bbox
        wpx = int(rx * slide_w); wpy = int(ry * slide_h)
        wpw = int(rw * slide_w); wph = int(rh * slide_h)
        draw_border(overlay, wpx, wpy, wpw, wph, line_width=2)

        # 該当ノード部分にハイライト＋ボーダー
        nx, ny, nw, nh = node_bbox
        npx = int(nx * slide_w); npy = int(ny * slide_h)
        npw = int(nw * slide_w); nph = int(nh * slide_h)
        ncx = npx + npw // 2;   ncy = npy + nph // 2

        if style in ("highlight", "all"):
            draw_highlight(overlay, npx, npy, npw, nph)
        if style in ("border", "all"):
            draw_border(overlay, npx, npy, npw, nph, line_width=BORDER_WIDTH)
        if style in ("laser", "all"):
            draw_laser(overlay, ncx, ncy, slide_w)
        if label:
            draw_label(overlay, npx, npy, label)

    else:
        # --- 通常要素 or SmartArt全体 ---
        rx, ry, rw, rh = whole_bbox
        px = int(rx * slide_w); py = int(ry * slide_h)
        pw = int(rw * slide_w); ph = int(rh * slide_h)
        cx = px + pw // 2;     cy = py + ph // 2

        if style in ("highlight", "all"):
            draw_highlight(overlay, px, py, pw, ph)
        if style in ("border", "all"):
            draw_border(overlay, px, py, pw, ph)
        if style in ("laser", "all"):
            draw_laser(overlay, cx, cy, slide_w)
        if label:
            draw_label(overlay, px, py, label)

    result = Image.alpha_composite(overlay_img, overlay).convert("RGB")

    if not output_path:
        stem = Path(slide_image_path).stem
        output_path = str(Path(slide_image_path).parent / f"{stem}_pointed.png")

    result.save(output_path, "PNG")
    print(f"[DRAW] 保存完了: {output_path}")
    return output_path


# ============================================================
# CLIエントリポイント
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="スライド画像にポインターを描画する"
    )
    parser.add_argument("slide_image", help="入力スライド画像のパス（PNG/JPG）")
    parser.add_argument(
        "--style", default="all",
        choices=["laser", "highlight", "border", "all"],
        help="ポインタースタイル（デフォルト: all）"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="出力画像のパス（省略時は入力ファイル名_pointed.png）"
    )
    parser.add_argument(
        "--no-label", action="store_true",
        help="ラベルテキストを非表示にする"
    )

    # bbox指定方法（どちらか一方）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bbox-ratio", nargs=4, type=float,
        metavar=("X", "Y", "W", "H"),
        help="bbox比率を直接指定（例: 0.078 0.187 0.854 0.739）"
    )
    group.add_argument(
        "--result-json",
        help="semantic_pointer.py の出力JSONファイルのパス"
    )
    group.add_argument(
        "--interactive", action="store_true",
        help="対話モード（semantic_pointer と統合テスト）"
    )

    # 対話モード用
    parser.add_argument("--slide-json", default=None,
                        help="[対話モード] slide_parser.py の出力JSONパス")
    parser.add_argument("--slide-num", type=int, default=1,
                        help="[対話モード] 現在のスライド番号（1-indexed）")
    parser.add_argument("--model", default="llama3",
                        help="[対話モード] Ollamaモデル名")

    args = parser.parse_args()

    if not os.path.exists(args.slide_image):
        sys.exit(f"[ERROR] 画像が見つかりません: {args.slide_image}")

    show_label = not args.no_label

    # --- モード1: bbox_ratio直接指定 ---
    if args.bbox_ratio:
        out = apply_pointer(
            args.slide_image, args.bbox_ratio,
            style=args.style, output_path=args.output,
            label="[manual]" if show_label else None
        )
        print(f"[DRAW] 保存完了: {out}")

    # --- モード2: JSONファイルから読み込み ---
    elif args.result_json:
        if not os.path.exists(args.result_json):
            sys.exit(f"[ERROR] JSONが見つかりません: {args.result_json}")
        with open(args.result_json, encoding="utf-8") as f:
            result = json.load(f)
        apply_from_result(
            args.slide_image, result,
            style=args.style, show_label=show_label,
            output_path=args.output
        )

    # --- モード3: 対話モード（semantic_pointer と統合） ---
    elif args.interactive:
        if not args.slide_json:
            sys.exit("[ERROR] 対話モードには --slide-json が必要です")

        # semantic_pointer をインポート
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from semantic_pointer import (
                load_slide_data, get_elements_for_slide, get_pointing_target
            )
        except ImportError:
            sys.exit("[ERROR] semantic_pointer.py が見つかりません。同じディレクトリに置いてください。")

        slide_data = load_slide_data(args.slide_json)
        elements   = get_elements_for_slide(slide_data, args.slide_num)

        if not elements:
            sys.exit(f"[ERROR] スライド {args.slide_num} の要素が見つかりません")

        print(f"[INFO] スライド {args.slide_num} / 要素数: {len(elements)}")
        print(f"[INFO] モデル: {args.model} / スタイル: {args.style}")
        print("\n対話モード: テキストを入力してEnter (終了: q)\n")

        counter = 0
        try:
            while True:
                text = input("ASRテキスト> ").strip()
                if text.lower() in ("q", "quit", "exit"):
                    break
                if not text:
                    continue

                result = get_pointing_target(text, elements, model=args.model)
                print(json.dumps(result, ensure_ascii=False, indent=2))

                if result.get("confidence") != "none":
                    counter += 1
                    stem = Path(args.slide_image).stem
                    out_path = str(
                        Path(args.slide_image).parent /
                        f"{stem}_pointed_{counter:03d}.png"
                    )
                    apply_from_result(
                        args.slide_image, result,
                        style=args.style, show_label=show_label,
                        output_path=out_path
                    )

        except KeyboardInterrupt:
            print("\n終了します。")


if __name__ == "__main__":
    main()
