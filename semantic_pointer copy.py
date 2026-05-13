"""
semantic_pointer.py
===================
ASRテキスト（音声認識結果）とスライド要素メタデータ（slide_parser.pyの出力JSON）を
受け取り、Ollama上のLlama3を使って「今どの要素をポインティングすべきか」を推論する
意味理解モジュール。

処理フロー:
    1. スライドJSONを読み込み、現在のスライド番号の要素リストを取得
    2. ASRテキスト1文を受け取る
    3. Llama3に「どの要素が最も関連するか」を推論させる
    4. 要素ID・bbox・理由をdictで返す

出力形式:
    {
        "element_id": "s2_e1",
        "element_type": "text",
        "content_preview": "意味的理解の表現",
        "bbox_pt": [74.88, 101.12, 820.36, 399.18],
        "bbox_ratio": [0.078, 0.1873, 0.8545, 0.7392],
        "reason": "The spoken text mentions semantic understanding...",
        "confidence": "high"   # high / medium / low / none
    }
    confidence が "none" の場合はポインティング不要と判断。

使い方（CLI）:
    # 対話モードで動作確認
    python semantic_pointer.py --slide-json slide_data.json --slide-num 5

    # 1文を直接渡す
    python semantic_pointer.py --slide-json slide_data.json --slide-num 5 \\
        --text "意味的な理解が重要です"
"""

import argparse
import json
import os
import sys
from typing import Optional
import re

# フィラー・スライド遷移フレーズの事前フィルタ
_TRANSITION_PATTERN = re.compile(
    r"次の?スライド|前の?スライド|スライドを?移|以上です|では次に"
    r"|^(えーと|あのー|そうですね|はい|うん)[、。\s]*$",
    re.IGNORECASE,
)
# --- Ollama ---
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] ollama ライブラリが見つかりません: pip install ollama")

# ============================================================
# 設定
# ============================================================
LLM_MODEL = "llama3"

# システムプロンプト（英語で指示する方がLlama3の精度が高い）
SYSTEM_PROMPT = """You are an AI assistant for a presentation auto-pointing system.
Your job is to analyze a speaker's transcribed speech and identify which element
on the current slide they are most likely referring to.

You will receive:
1. The speaker's transcribed text (in Japanese)
2. A list of slide elements with their IDs, types, and text content

You must respond ONLY with a valid JSON object in this exact format:
{
  "element_id": "<id of the most relevant element, or null if none>",
  "confidence": "<high|medium|low|none>",
  "reason": "<brief explanation in English>",
  "smartart_node_index": <0-based index of the matching node if type is smartart, otherwise null>
}

Rules:
- Choose "none" confidence if the speech is a transition phrase, filler, or unrelated to any element
- Choose the single MOST relevant element even if multiple could match
- For smartart elements, the content is a list of nodes — identify the single most relevant node index (0-based)
- SmartArt nodes often come in pairs: even-indexed nodes (0, 2, 4...) are headings, odd-indexed nodes (1, 3, 5...) are descriptions. If the speech matches a heading, return the heading's index (even number).
- For smartart elements, you MUST always return a specific smartart_node_index (never null). Even if the speech matches the whole smartart, choose the most relevant node index.
- Do NOT include any text outside the JSON object
"""

def build_user_prompt(asr_text: str, elements: list[dict]) -> str:
    """LLMに渡すユーザープロンプトを組み立てる。"""
    # 要素リストを簡潔なテキストに変換
    elem_lines = []
    for e in elements:
        eid = e["id"]
        etype = e["type"]
        content = e.get("content")
        vlm_desc = e.get("vlm_description")

        if isinstance(content, list):
            # SmartArtのノードリスト: インデックス付きで表示してLLMが番号を返せるように
            content_str = " | ".join(f"[{i}]{t}" for i, t in enumerate(content))
            # SmartArtには必ずnode_indexを選ぶよう明示的に指示を追加
            content_str += f"  ※ MUST return smartart_node_index (0 to {len(content)-1})"
        elif content:
            content_str = content[:60]  # 長すぎる場合は切り詰め
        elif vlm_desc:
            content_str = f"[image] {vlm_desc[:60]}"
        else:
            content_str = "(no text content)"

        elem_lines.append(f'  - id="{eid}" type="{etype}": {content_str}')

    elements_text = "\n".join(elem_lines)
    valid_ids = ", ".join(f'"{e["id"]}"' for e in elements)

    return f"""Speaker's transcribed text (Japanese):
\"{asr_text}\"

Slide elements on the current slide:
{elements_text}

Which element is the speaker most likely referring to?
IMPORTANT: You MUST use ONLY one of these exact element IDs: {valid_ids}
Do NOT invent, combine, or modify IDs. If unsure, pick the closest match from the list.
Respond with JSON only."""


# ============================================================
# メイン推論関数
# ============================================================

def get_pointing_target(
    asr_text: str,
    slide_elements: list[dict],
    model: str = LLM_MODEL,
    verbose: bool = False
) -> dict:
    """
    ASRテキストとスライド要素リストからポインティング対象を推論する。

    Parameters
    ----------
    asr_text : str
        音声認識で得られたテキスト（1文）
    slide_elements : list[dict]
        slide_parser.py が出力したスライド要素のリスト
    model : str
        使用するOllamaモデル名
    verbose : bool
        Trueの場合、プロンプトとLLM生レスポンスを表示する

    Returns
    -------
    dict
        ポインティング結果
    """
    if _TRANSITION_PATTERN.search(asr_text):
        return _no_target_result("Filtered by transition/filler pattern before LLM")

    if not OLLAMA_AVAILABLE:
        return _error_result("ollama not installed")

    # タイトル要素（スライドタイトル）はポインティング対象から除外
    pointable = [e for e in slide_elements
                 if not (e["type"] == "text" and
                         ("タイトル" in e.get("shape_name", "") or
                          "Title" in e.get("shape_name", "")))]

    if not pointable:
        return _no_target_result("No pointable elements on this slide")

    user_prompt = build_user_prompt(asr_text, pointable)

    if verbose:
        print("\n--- [LLM] User Prompt ---")
        print(user_prompt)

    try:
        response = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            options={
                "num_predict": 256,  # JSONが途中で切れないよう十分な長さを確保
                "temperature": 0.1,  # 出力を安定させる
            }
        )
        raw = response["message"]["content"].strip()

        if verbose:
            print(f"\n--- [LLM] Raw Response ---\n{raw}")

        # JSONパース（LLMがコードブロックで囲む場合があるので除去）
        clean = raw
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        # 閉じ括弧が欠けている場合に補完を試みる
        if clean and not clean.endswith("}"):
            last_comma = clean.rfind(",")
            last_close_quote = clean.rfind('"')
            if last_close_quote > last_comma:
                clean = clean[:last_close_quote + 1] + "\n}"
            elif last_comma > 0:
                clean = clean[:last_comma] + "\n}"
            else:
                clean = clean + "\n}"

        llm_result = json.loads(clean)

    except json.JSONDecodeError as e:
        return _error_result(f"JSON parse error: {e} / raw: {raw[:200]}")
    except Exception as e:
        return _error_result(str(e))

    # LLM結果を元に要素の詳細情報を付加
    element_id = llm_result.get("element_id")
    confidence  = llm_result.get("confidence", "none")
    reason      = llm_result.get("reason", "")

    if not element_id or confidence == "none":
        return _no_target_result(reason)

    # 対応する要素をリストから探す
    matched = next((e for e in slide_elements if e["id"] == element_id), None)
    if not matched:
        # LLMが存在しないIDを返した場合のフォールバック:
        # SmartArt要素を優先して代替する（タイトル等は除外）
        slide_prefix = element_id.rsplit("_", 1)[0]  # 例: "s4_e2" → "s4"
        candidates = [e for e in slide_elements
                      if e["id"].startswith(slide_prefix)
                      and e["type"] == "smartart"]
        if not candidates:
            # SmartArtがなければtextも候補に（ただしタイトルは除外）
            candidates = [e for e in slide_elements
                          if e["id"].startswith(slide_prefix)
                          and e["type"] != "text"
                          or (e["type"] == "text"
                              and "タイトル" not in e.get("shape_name", "")
                              and "Title" not in e.get("shape_name", ""))]
        if candidates:
            matched = candidates[0]
            reason = f"[FALLBACK] LLM returned unknown id '{element_id}', using '{matched['id']}' instead. " + reason
        else:
            return _error_result(f"element_id '{element_id}' not found in slide elements")

    # contentのプレビュー生成
    content = matched.get("content")
    if isinstance(content, list):
        preview = " / ".join(content)[:80]
    elif content:
        preview = content[:80]
    else:
        preview = matched.get("vlm_description", "")[:80] or "(no content)"

    # SmartArtの場合はノードインデックスからbboxを等分割して計算
    smartart_node_index    = llm_result.get("smartart_node_index")
    smartart_node_bbox_ratio = None

    if matched["type"] == "smartart" and smartart_node_index is not None:
        node_count = matched.get("smartart_node_count", 1)
        if node_count > 0 and isinstance(smartart_node_index, int):
            rx, ry, rw, rh = matched["bbox_ratio"]

            # ペア構造の検出: 偶数ノード=見出し、奇数ノード=説明文のペアかどうか
            # ノード数が偶数の場合はペア構造と見なし、グループ数で等分割する
            # 上下パディングを除いた有効領域で計算（SmartArtの余白を補正）
            PADDING = 0.04  # SmartArt上下の余白比率（調整可能）
            effective_ry = ry + rh * PADDING
            effective_rh = rh * (1 - PADDING * 2)

            if node_count % 2 == 0:
                # ペア構造: (見出し+説明文) を1グループとして扱う
                group_count = node_count // 2
                group_index = smartart_node_index // 2  # 偶数奇数どちらでも同じグループに
                group_h = effective_rh / group_count
                group_ry = effective_ry + group_h * group_index
                smartart_node_bbox_ratio = [
                    round(rx, 4),
                    round(group_ry, 4),
                    round(rw, 4),
                    round(group_h, 4),
                ]
            else:
                # 奇数ノード: 通常の等分割
                node_h = effective_rh / node_count
                node_ry = effective_ry + node_h * smartart_node_index
                smartart_node_bbox_ratio = [
                    round(rx, 4),
                    round(node_ry, 4),
                    round(rw, 4),
                    round(node_h, 4),
                ]

    return {
        "element_id":              element_id,
        "element_type":            matched["type"],
        "content_preview":         preview,
        "bbox_pt":                 matched["bbox_pt"],
        "bbox_ratio":              matched["bbox_ratio"],
        "shape_name":              matched.get("shape_name", ""),
        "reason":                  reason,
        "confidence":              confidence,
        "smartart_node_index":     smartart_node_index,
        "smartart_node_bbox_ratio": smartart_node_bbox_ratio,
        "chart_pointing":           None,  # server.pyで非同期で別送信
        "is_image_element":         matched["type"] == "image" and bool(matched.get("vlm_description")),  # ← 追加
    }
# ============================================================
# 図表内部へのセマンティックポインティング（4.2）
# ============================================================

_CHART_POINTING_SYSTEM = """You are an AI assistant for a presentation auto-pointing system.
Given a speaker's utterance and a description of a chart/graph/diagram,
determine what visual annotation should be drawn on the chart.

IMPORTANT: Read the chart description carefully to identify the ORDER of bars from left to right.
Map each bar name to its position index (0=leftmost, 1=middle, 2=rightmost, etc.)

For a 3-bar chart:
  Bar index 0 (leftmost) : x_left=0.05, x_right=0.28
  Bar index 1 (middle)   : x_left=0.37, x_right=0.60
  Bar index 2 (rightmost): x_left=0.69, x_right=0.92

For y-coordinates:
  y_bottom = 0.75
  Tallest bar:  y_top ≈ 0.05
  Medium bar:   y_top ≈ 0.37
  Shortest bar: y_top ≈ 0.46

Steps to determine bbox_ratio:
  1. Find the bar name mentioned in the utterance
  2. Look up its position index from the chart description (left-to-right order)
  3. Use the x coordinates for that index
  4. Use y coordinates based on bar height

You must respond ONLY with a valid JSON object in this exact format:
{
  "draw_type": "<circle|arrow|none>",
  "target_description": "<brief description>",
  "bbox_ratio": [x_left, y_top, width, height],
  "arrow_start_ratio": [x, y],
  "arrow_end_ratio": [x, y],
  "reason": "<which bar index was selected and why>"
}

Rules:
- draw_type "circle": speaker refers to a specific bar or data point
- draw_type "arrow": speaker refers to a trend (increasing, decreasing, etc.)
  → arrow_start_ratio = [leftmost_center_x, 0.4], arrow_end_ratio = [rightmost_center_x, 0.4]
- draw_type "none": utterance refers to the chart as a whole
- All coordinates relative to the chart element's bounding box (0,0=top-left, 1,1=bottom-right)
- Do NOT include any text outside the JSON object
"""

def get_chart_pointing(
    asr_text: str,
    vlm_description: str,
    image_bbox_ratio: list[float],
    image_b64: str | None = None, 
    model: str = "llama3",
    verbose: bool = False,
) -> dict | None:
    """
    図表要素が選ばれたときに、図表内部のどこを指すかを推論する。

    Parameters
    ----------
    asr_text : str
        発話テキスト
    vlm_description : str
        slide_parser.pyが生成した図表の説明文
    image_bbox_ratio : list[float]
        図表要素全体のbbox_ratio [x, y, w, h]（スライド全体に対する比率）
    model : str
        使用するOllamaモデル名

    Returns
    -------
    dict | None
        {
          draw_type: "circle" | "arrow" | "none",
          abs_bbox_ratio: [x, y, w, h] | None,   # スライド全体座標系に変換済み
          abs_arrow_start: [x, y] | None,
          abs_arrow_end:   [x, y] | None,
          reason: str,
        }
        推論失敗時は None を返す。
    """
    if not OLLAMA_AVAILABLE:
        return None

    user_prompt = f"""Speaker's utterance (Japanese):
"{asr_text}"

Chart/diagram description:
{vlm_description[:600]}

Based on the utterance, what should be annotated on this chart?
Respond with JSON only."""

    if verbose:
        print("\n--- [CHART_POINTING] User Prompt ---")
        print(user_prompt[:300])

    message_content = {
        "role": "user",
        "content": user_prompt,
    }
    if image_b64:
        message_content["images"] = [image_b64]
    try:
        response = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _CHART_POINTING_SYSTEM},
                message_content,
            ],
            options={"num_predict": 1024, "temperature": 0.1, "think": False},
        )
        raw = response["message"]["content"].strip()
        thinking = response["message"].thinking or ""

        print(f"    [CHART_POINTING DEBUG] content_len={len(raw)} thinking_len={len(thinking)}")
        print(f"    [CHART_POINTING DEBUG] thinking_tail={repr(thinking[-200:])}")
        # think=Falseが効かない場合のフォールバック: thinkingフィールドを使う
        if not raw:
            thinking = response["message"].thinking or ""
            # thinkingの中からJSONブロックを探す
            import re as _re
            json_match = _re.search(r'\{[^{}]*"draw_type"[^{}]*\}', thinking, _re.DOTALL)
            if json_match:
                raw = json_match.group(0)
                print(f"    [CHART_POINTING] thinkingからJSON抽出: {raw[:80]}")
            else:
                # thinkingの末尾にJSONがある場合
                last_brace = thinking.rfind("}")
                first_brace = thinking.rfind("{", 0, last_brace)
                if first_brace != -1 and last_brace != -1:
                    raw = thinking[first_brace:last_brace+1]
                    print(f"    [CHART_POINTING] thinkingから末尾JSON抽出: {raw[:80]}")

        print(f"    [CHART_POINTING DEBUG] raw={repr(raw[:150])}")
        if verbose:
            print(f"\n--- [CHART_POINTING] Raw ---\n{raw}")

        # JSONパース
        clean = raw
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
        if clean and not clean.endswith("}"):
            clean = clean[:clean.rfind("}") + 1] if "}" in clean else clean + "}"

        parsed = json.loads(clean)

    except Exception as e:
        print(f"    [CHART_POINTING] エラー: {e}")
        return None

    draw_type = parsed.get("draw_type", "none")
    if draw_type == "none":
        return {"draw_type": "none", "abs_bbox_ratio": None,
                "abs_arrow_start": None, "abs_arrow_end": None,
                "reason": parsed.get("reason", "")}

    # 図表内部の相対座標 → スライド全体の絶対座標に変換
    ix, iy, iw, ih = image_bbox_ratio

    abs_bbox_ratio    = None
    abs_arrow_start   = None
    abs_arrow_end     = None

    if draw_type == "circle":
        local_bbox = parsed.get("bbox_ratio")
        if local_bbox and len(local_bbox) == 4:
            lx, ly, lw, lh = local_bbox
            abs_bbox_ratio = [
                round(ix + lx * iw, 4),
                round(iy + ly * ih, 4),
                round(lw * iw, 4),
                round(lh * ih, 4),
            ]

    elif draw_type == "arrow":
        start = parsed.get("arrow_start_ratio")
        end   = parsed.get("arrow_end_ratio")
        if start and end and len(start) == 2 and len(end) == 2:
            abs_arrow_start = [
                round(ix + start[0] * iw, 4),
                round(iy + start[1] * ih, 4),
            ]
            abs_arrow_end = [
                round(ix + end[0] * iw, 4),
                round(iy + end[1] * ih, 4),
            ]

    result = {
        "draw_type":       draw_type,
        "abs_bbox_ratio":  abs_bbox_ratio,
        "abs_arrow_start": abs_arrow_start,
        "abs_arrow_end":   abs_arrow_end,
        "reason":          parsed.get("reason", ""),
        "target":          parsed.get("target_description", ""),
    }

    print(f"    [CHART_POINTING] draw_type={draw_type}  "
          f"bbox={abs_bbox_ratio}  "
          f"start={abs_arrow_start} end={abs_arrow_end}")

    return result

def _no_target_result(reason: str) -> dict:
    """ポインティング不要の結果を返す。"""
    return {
        "element_id":      None,
        "element_type":    None,
        "content_preview": None,
        "bbox_pt":         None,
        "bbox_ratio":      None,
        "shape_name":      None,
        "reason":          reason,
        "confidence":      "none",
    }


def _error_result(msg: str) -> dict:
    """エラー時の結果を返す。"""
    return {
        "element_id":      None,
        "element_type":    None,
        "content_preview": None,
        "bbox_pt":         None,
        "bbox_ratio":      None,
        "shape_name":      None,
        "reason":          f"[ERROR] {msg}",
        "confidence":      "none",
    }


# ============================================================
# スライドJSON読み込みユーティリティ
# ============================================================

def load_slide_data(json_path: str) -> dict:
    """スライドJSONを読み込む。"""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_elements_for_slide(slide_data: dict, slide_number: int) -> list[dict]:
    """指定スライド番号の要素リストを返す（1-indexed）。"""
    for s in slide_data["slides"]:
        if s["slide_number"] == slide_number:
            return s["elements"]
    return []


# ============================================================
# CLIエントリポイント
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="意味理解モジュール - ASRテキストからポインティング対象を推論"
    )
    parser.add_argument("--slide-json", required=True,
                        help="slide_parser.py が出力したJSONファイルのパス")
    parser.add_argument("--slide-num", type=int, required=True,
                        help="現在表示中のスライド番号 (1-indexed)")
    parser.add_argument("--text", default=None,
                        help="ASRテキスト（省略時は対話モード）")
    parser.add_argument("--model", default=LLM_MODEL,
                        help=f"使用するOllamaモデル (デフォルト: {LLM_MODEL})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="LLMのプロンプトと生レスポンスを表示する")
    args = parser.parse_args()

    if not os.path.exists(args.slide_json):
        sys.exit(f"[ERROR] JSONファイルが見つかりません: {args.slide_json}")

    slide_data = load_slide_data(args.slide_json)
    elements = get_elements_for_slide(slide_data, args.slide_num)

    if not elements:
        sys.exit(f"[ERROR] スライド {args.slide_num} の要素が見つかりません")

    print(f"[INFO] スライド {args.slide_num} の要素数: {len(elements)}")
    print(f"[INFO] モデル: {args.model}")

    def run_once(text: str):
        print(f"\n>>> ASR: {text}")
        result = get_pointing_target(
            text, elements, model=args.model, verbose=args.verbose)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.text:
        # 1文モード
        run_once(args.text)
    else:
        # 対話モード
        print("\n対話モード: テキストを入力してEnter (終了: Ctrl+C or 'q')\n")
        try:
            while True:
                text = input("ASRテキスト> ").strip()
                if text.lower() in ("q", "quit", "exit"):
                    break
                if not text:
                    continue
                run_once(text)
        except KeyboardInterrupt:
            print("\n終了します。")


if __name__ == "__main__":
    main()
