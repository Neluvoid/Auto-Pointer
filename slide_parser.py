"""
slide_parser.py
===============
PPTXファイルを解析し、各スライドの要素（テキスト・図形・画像）を
座標情報とともに抽出するモジュール。
画像・図表はOllama上のMoondream2で内容解析を行う。

出力形式 (JSON):
{
  "file": "xxx.pptx",
  "slide_width_pt": 720.0,
  "slide_height_pt": 540.0,
  "slides": [
    {
      "slide_index": 0,
      "slide_number": 1,
      "elements": [
        {
          "id": "s0_e0",
          "type": "text",          # text / image / shape / table / smartart
          "content": "タイトル",
          "bbox_pt": [left, top, width, height],   # ポイント単位
          "bbox_ratio": [x, y, w, h],              # スライドサイズに対する比率 (0~1)
          "shape_name": "Title 1"
        },
        {
          "id": "s0_e1",
          "type": "image",
          "content": null,
          "vlm_description": "A bar chart showing...",  # Moondream2による解析結果
          "bbox_pt": [...],
          "bbox_ratio": [...],
          "shape_name": "Picture 3"
        }
      ]
    }
  ]
}

使い方:
    python slide_parser.py your_slide.pptx [--no-vlm] [--output result.json]
"""

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

# --- 依存ライブラリ ---
try:
    from pptx import Presentation
    from pptx.util import Pt
    from pptx.enum.shapes import MSO_SHAPE_TYPE
except ImportError:
    sys.exit("[ERROR] python-pptx が見つかりません。'pip install python-pptx' を実行してください。")

try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Pillow が見つかりません。'pip install Pillow' を実行してください。")

# Ollamaはオプション（--no-vlm 指定時は不要）
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# COMオートメーション（Windows + PowerPointがある場合のみ）
try:
    import comtypes.client
    COM_AVAILABLE = True
except ImportError:
    COM_AVAILABLE = False


# ============================================================
# 設定
# ============================================================
VLM_MODEL = "moondream"          # Ollama上のVLMモデル名
VLM_PROMPT = (
    "Describe all visual elements in this image in detail. "
    "If it is a chart or graph, identify the chart type, axes labels, "
    "legend items, and notable data points or trends. "
    "If it is a diagram, describe the components and their relationships. "
    "Respond in English."
)


# ============================================================
# ユーティリティ
# ============================================================

def emu_to_pt(emu: int) -> float:
    """EMU (English Metric Units) をポイントに変換する。1pt = 12700 EMU"""
    return emu / 12700.0


def bbox_to_ratio(left_pt, top_pt, width_pt, height_pt,
                  slide_width_pt, slide_height_pt) -> list[float]:
    """ポイント座標をスライドサイズ比率 (0~1) に変換する。"""
    return [
        round(left_pt / slide_width_pt, 4),
        round(top_pt / slide_height_pt, 4),
        round(width_pt / slide_width_pt, 4),
        round(height_pt / slide_height_pt, 4),
    ]


def pil_image_to_base64(img: "Image.Image", fmt: str = "PNG") -> str:
    """PIL Imageをbase64文字列に変換する。"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# VLM解析
# ============================================================

def analyze_image_with_vlm(img: "Image.Image", model: str = VLM_MODEL) -> str:
    """
    PIL ImageをMoondream2（Ollama）に渡し、内容の説明文を返す。
    Ollamaが利用できない場合や失敗した場合は空文字列を返す。
    """
    if not OLLAMA_AVAILABLE:
        return "[VLM unavailable: ollama not installed]"
    try:
        img_b64 = pil_image_to_base64(img)
        response = _ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": VLM_PROMPT,
                "images": [img_b64]
            }]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[VLM error: {e}]"


# ============================================================
# PowerPoint COMオートメーション：スライド画像エクスポート
# ============================================================

def export_slides_as_images(pptx_path: str,
                             output_dir: str,
                             width_px: int = 1920) -> list[str]:
    """
    PowerPoint COMオートメーションを使ってPPTXの全スライドをPNG画像として出力する。
    Windows + PowerPointがインストールされている環境でのみ動作する。

    Parameters
    ----------
    pptx_path : str
        入力PPTXファイルのパス
    output_dir : str
        画像の出力先ディレクトリ
    width_px : int
        出力画像の横幅（ピクセル）。縦はアスペクト比から自動計算。

    Returns
    -------
    list[str]
        出力された画像ファイルのパスリスト（スライド順）
    """
    if not COM_AVAILABLE:
        raise RuntimeError(
            "comtypes が見つかりません。'pip install comtypes' を実行してください。"
        )

    os.makedirs(output_dir, exist_ok=True)
    abs_pptx = os.path.abspath(pptx_path)
    abs_out  = os.path.abspath(output_dir)

    print(f"[COM] PowerPoint を起動してスライド画像を生成中...")
    ppt_app = None
    try:
        ppt_app = comtypes.client.CreateObject("PowerPoint.Application")
        ppt_app.Visible = 1  # 1 = msoTrue

        prs = ppt_app.Presentations.Open(abs_pptx, ReadOnly=1, WithWindow=0)

        slide_w = prs.PageSetup.SlideWidth   # ポイント単位
        slide_h = prs.PageSetup.SlideHeight
        height_px = int(width_px * slide_h / slide_w)

        image_paths = []
        for i, slide in enumerate(prs.Slides):
            out_path = os.path.join(abs_out, f"slide_{i+1:03d}.png")
            slide.Export(out_path, "PNG", width_px, height_px)
            image_paths.append(out_path)
            print(f"[COM]   スライド {i+1}/{prs.Slides.Count} -> {out_path}")

        prs.Close()
        return image_paths

    finally:
        if ppt_app is not None:
            ppt_app.Quit()
            print("[COM] PowerPoint を終了しました。")


# ============================================================
# SmartArt解析
# ============================================================

# SmartArtのgraphicData URI
_SMARTART_URI = "http://schemas.openxmlformats.org/drawingml/2006/diagram"

def is_smartart(shape) -> bool:
    """graphicFrame要素がSmartArtかどうかを判定する。"""
    try:
        xml_str = shape._element.xml
        return _SMARTART_URI in xml_str
    except Exception:
        return False


def extract_smartart_texts(pptx_path: str, slide_idx: int, shape_name: str) -> list[str]:
    """
    PPTXのZIPからSmartArtのdataXMLを読み、テキストノードを抽出する。
    shape_nameを使って対応するdataファイルを特定する。
    """
    import zipfile
    import re

    texts = []
    try:
        with zipfile.ZipFile(pptx_path, 'r') as z:
            # diagrams/data*.xml を全部読んでテキスト抽出
            data_files = sorted([f for f in z.namelist()
                                  if re.match(r'ppt/diagrams/data\d+\.xml', f)])
            # スライドのrelファイルからどのdataXMLが対応するか特定
            rel_path = f'ppt/slides/_rels/slide{slide_idx + 1}.xml.rels'
            diagram_data_file = None
            if rel_path in z.namelist():
                rels_content = z.read(rel_path).decode('utf-8')
                # diagram dataへの参照を探す
                matches = re.findall(
                    r'Target=\"\.\./diagrams/(data\d+\.xml)\"', rels_content)
                if matches:
                    diagram_data_file = f'ppt/diagrams/{matches[0]}'

            if diagram_data_file and diagram_data_file in z.namelist():
                content = z.read(diagram_data_file).decode('utf-8')
                texts = re.findall(r'<a:t[^>]*>([^<]+)</a:t>', content)
            elif data_files:
                # fallback: 全dataファイルからテキスト抽出
                for df in data_files:
                    content = z.read(df).decode('utf-8')
                    found = re.findall(r'<a:t[^>]*>([^<]+)</a:t>', content)
                    texts.extend(found)
    except Exception as e:
        texts = [f"[SmartArt parse error: {e}]"]

    return texts


def render_smartart_as_image(pptx_path: str, slide_idx: int,
                              bbox_pt: list, slide_size_pt: tuple) -> Optional["Image.Image"]:
    """
    SmartArtをスライド画像としてレンダリングし、
    bbox領域をクロップしたPIL Imageを返す。
    LibreOfficeがインストールされていれば使用、なければNoneを返す。
    """
    import subprocess
    import tempfile
    import shutil

    if shutil.which("libreoffice") is None and shutil.which("soffice") is None:
        return None

    cmd = shutil.which("libreoffice") or shutil.which("soffice")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # PPTXをPNG画像に変換
            subprocess.run(
                [cmd, "--headless", "--convert-to", "png",
                 "--outdir", tmpdir, pptx_path],
                capture_output=True, timeout=60
            )
            # 変換されたPNGファイルを探す
            stem = Path(pptx_path).stem
            png_files = sorted(Path(tmpdir).glob(f"{stem}*.png"))
            if slide_idx >= len(png_files):
                return None
            slide_img = Image.open(png_files[slide_idx]).convert("RGB")

            # スライド画像サイズとポイント座標のスケール計算
            img_w, img_h = slide_img.size
            sw_pt, sh_pt = slide_size_pt
            scale_x = img_w / sw_pt
            scale_y = img_h / sh_pt

            left, top, width, height = bbox_pt
            crop_box = (
                int(left * scale_x),
                int(top * scale_y),
                int((left + width) * scale_x),
                int((top + height) * scale_y),
            )
            return slide_img.crop(crop_box)
    except Exception:
        return None


# ============================================================
# 図形タイプ判定
# ============================================================

def get_shape_type_str(shape) -> str:
    """python-pptx の Shape オブジェクトから要素タイプ文字列を返す。"""
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        return "image"
    if shape.has_text_frame:
        return "text"
    if shape.has_table:
        return "table"
    # graphicFrame = SmartArt or Chart
    if is_smartart(shape):
        return "smartart"
    # グループ・コネクタ・その他図形
    return "shape"


# ============================================================
# テキスト抽出
# ============================================================

def extract_text_content(shape) -> str:
    """テキストフレームを持つ図形からテキストを抽出する。"""
    lines = []
    for para in shape.text_frame.paragraphs:
        line = "".join(run.text for run in para.runs).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_table_content(shape) -> str:
    """テーブル図形からセルテキストをCSV風に抽出する。"""
    rows = []
    for row in shape.table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


# ============================================================
# 画像抽出
# ============================================================

def extract_image_from_shape(shape) -> Optional["Image.Image"]:
    """PICTURE図形から PIL Image を取得する。失敗時は None を返す。"""
    try:
        image_blob = shape.image.blob
        return Image.open(io.BytesIO(image_blob)).convert("RGB")
    except Exception:
        return None


# ============================================================
# メイン解析関数
# ============================================================

def parse_pptx(pptx_path: str, use_vlm: bool = True) -> dict:
    """
    PPTXファイルを解析し、スライド要素メタデータの辞書を返す。

    Parameters
    ----------
    pptx_path : str
        入力PPTXファイルのパス
    use_vlm : bool
        True の場合、画像・SmartArt要素をMoondream2で解析する

    Returns
    -------
    dict
        スライド要素メタデータ
    """
    prs = Presentation(pptx_path)

    slide_width_pt = emu_to_pt(prs.slide_width)
    slide_height_pt = emu_to_pt(prs.slide_height)
    slide_size_pt = (slide_width_pt, slide_height_pt)

    result = {
        "file": os.path.basename(pptx_path),
        "slide_width_pt": round(slide_width_pt, 2),
        "slide_height_pt": round(slide_height_pt, 2),
        "slides": []
    }

    total_slides = len(prs.slides)
    for slide_idx, slide in enumerate(prs.slides):
        print(f"  Processing slide {slide_idx + 1}/{total_slides}...")

        slide_data = {
            "slide_index": slide_idx,
            "slide_number": slide_idx + 1,
            "elements": []
        }

        for elem_idx, shape in enumerate(slide.shapes):
            elem_id = f"s{slide_idx}_e{elem_idx}"
            shape_type = get_shape_type_str(shape)

            # --- 座標情報 ---
            left_pt = emu_to_pt(shape.left) if shape.left is not None else 0.0
            top_pt = emu_to_pt(shape.top) if shape.top is not None else 0.0
            width_pt = emu_to_pt(shape.width) if shape.width is not None else 0.0
            height_pt = emu_to_pt(shape.height) if shape.height is not None else 0.0

            bbox_pt = [
                round(left_pt, 2),
                round(top_pt, 2),
                round(width_pt, 2),
                round(height_pt, 2)
            ]
            bbox_ratio = bbox_to_ratio(
                left_pt, top_pt, width_pt, height_pt,
                slide_width_pt, slide_height_pt
            )

            elem = {
                "id": elem_id,
                "type": shape_type,
                "content": None,
                "bbox_pt": bbox_pt,
                "bbox_ratio": bbox_ratio,
                "shape_name": shape.name,
            }

            # --- コンテンツ抽出 ---
            if shape_type == "text":
                elem["content"] = extract_text_content(shape)

            elif shape_type == "table":
                elem["content"] = extract_table_content(shape)

            elif shape_type == "image":
                pil_img = extract_image_from_shape(shape)
                if pil_img and use_vlm:
                    print(f"    [VLM] Analyzing image in element {elem_id}...")
                    elem["vlm_description"] = analyze_image_with_vlm(pil_img)
                else:
                    elem["vlm_description"] = None

            elif shape_type == "smartart":
                # ① ZIPからテキストノードを抽出（必ず実行）
                texts = extract_smartart_texts(pptx_path, slide_idx, shape.name)
                elem["content"] = texts  # リスト形式で保持
                elem["smartart_node_count"] = len(texts)
                print(f"    [SmartArt] slide{slide_idx+1} '{shape.name}': "
                      f"{len(texts)}ノード → {texts}")

                # ② VLM有効時: LibreOfficeでレンダリングしてVLM解析（オプション）
                elem["vlm_description"] = None
                if use_vlm:
                    print(f"    [VLM] Attempting SmartArt rendering for {elem_id}...")
                    cropped = render_smartart_as_image(
                        pptx_path, slide_idx, bbox_pt, slide_size_pt)
                    if cropped:
                        elem["vlm_description"] = analyze_image_with_vlm(cropped)
                        print(f"    [VLM] SmartArt VLM analysis complete.")
                    else:
                        print(f"    [VLM] LibreOffice not found, skipping render.")

            elif shape_type == "shape":
                # テキストなし図形でもテキストフレームを持つ場合がある
                if shape.has_text_frame:
                    text = extract_text_content(shape)
                    if text:
                        elem["content"] = text

            slide_data["elements"].append(elem)

        result["slides"].append(slide_data)

    return result


# ============================================================
# CLIエントリポイント
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PPTXスライド解析モジュール - 要素の座標とコンテンツをJSONで出力"
    )
    parser.add_argument("pptx_file", help="入力PPTXファイルのパス")
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="VLM（Moondream2）による画像解析をスキップする"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="出力JSONファイルのパス（省略時は入力ファイル名.jsonに保存）"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="JSONを整形して出力する（デフォルト: True）"
    )
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="PowerPoint COMでスライド画像を自動エクスポートする（Windows限定）"
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="画像出力先ディレクトリ（省略時はPPTXファイル名_imagesフォルダ）"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1920,
        help="出力画像の横幅px（デフォルト: 1920）"
    )
    args = parser.parse_args()

    pptx_path = args.pptx_file
    if not os.path.exists(pptx_path):
        sys.exit(f"[ERROR] ファイルが見つかりません: {pptx_path}")

    use_vlm = not args.no_vlm
    if use_vlm and not OLLAMA_AVAILABLE:
        print("[WARNING] ollama ライブラリが見つかりません。VLM解析をスキップします。")
        use_vlm = False

    print(f"[INFO] Parsing: {pptx_path}")
    print(f"[INFO] VLM解析: {'有効 (Moondream2)' if use_vlm else '無効'}")

    data = parse_pptx(pptx_path, use_vlm=use_vlm)

    # 出力パス決定
    if args.output:
        out_path = args.output
    else:
        out_path = str(Path(pptx_path).stem) + "_slide_data.json"

    indent = 2 if args.pretty else None
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    print(f"[INFO] 保存完了: {out_path}")

    # 画像エクスポート（オプション）
    if args.export_images:
        images_dir = args.images_dir or (str(Path(pptx_path).stem) + "_images")
        try:
            image_paths = export_slides_as_images(
                pptx_path, images_dir, width_px=args.image_width)
            print(f"[INFO] 画像エクスポート完了: {len(image_paths)}枚 -> {images_dir}/")
            # JSONにも画像パスを埋め込む
            for slide in data["slides"]:
                idx = slide["slide_index"]
                if idx < len(image_paths):
                    slide["image_path"] = image_paths[idx]
            # 画像パス付きで再保存
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            print(f"[INFO] JSONに画像パスを追記して再保存: {out_path}")
        except Exception as e:
            print(f"[WARNING] 画像エクスポート失敗: {e}")

    # サマリー表示
    total_elements = sum(len(s["elements"]) for s in data["slides"])
    print(f"\n=== 解析サマリー ===")
    print(f"  スライド数     : {len(data['slides'])}")
    print(f"  総要素数       : {total_elements}")
    for slide in data["slides"]:
        type_counts = {}
        for e in slide["elements"]:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        print(f"  スライド {slide['slide_number']:2d}    : {type_counts}")


if __name__ == "__main__":
    main()
