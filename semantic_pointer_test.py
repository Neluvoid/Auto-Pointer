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
    }


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
