"""
model_comparison.py
===================
llama3 vs gemma4:12b の精度・速度比較スクリプト（4.1 意味理解精度向上）

テスト方法:
  - 実際の semantic_pointer.py と同じシステムプロンプト・ユーザープロンプトを使用
  - 正解ラベル付きテストケースで精度を自動採点
  - 各モデルを各ケース N_RUNS 回実行して平均速度を計測

使い方:
    # 基本（llama3 vs gemma4:12b）
    python model_comparison.py

    # モデルを追加指定
    python model_comparison.py --models llama3 gemma4:12b llama3.1:8b

    # 自前のスライドJSONを使ってテストケース自動生成
    python model_comparison.py --slide-json your_slide_data.json --slide-num 3

    # 繰り返し回数を変更（速度計測精度向上）
    python model_comparison.py --runs 5

出力:
    - コンソールにサマリーテーブル
    - model_comparison_result.csv（詳細ログ）
    - model_comparison_summary.csv（モデル別サマリー）
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import re
_TRANSITION_PATTERN = re.compile(
    r"次の?スライド|前の?スライド|スライドを?移|以上です|では次に"
    r"|^(えーと|あのー|そうですね|はい|うん)[、。\s]*$",
    re.IGNORECASE,
)

try:
    import ollama as _ollama
except ImportError:
    sys.exit("[ERROR] ollama ライブラリが見つかりません: pip install ollama")

# ============================================================
# semantic_pointer.py と同じプロンプト定義
# ============================================================

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
- Choose "none" confidence ONLY when the speech clearly falls into one of these categories:
    * Slide transition phrases: "次のスライド", "前のスライド", "スライドを移ります", "以上です", "では次に"
    * Pure fillers with no content: "えーと", "あのー", "そうですね" (when spoken alone with nothing else)
    * Completely off-topic speech unrelated to any slide element
  If you are unsure whether to choose "none", prefer choosing the most relevant element instead.
- Choose the single MOST relevant element even if multiple could match
"""

def build_user_prompt(asr_text: str, elements: list[dict]) -> str:
    elem_lines = []
    for e in elements:
        eid     = e["id"]
        etype   = e["type"]
        content = e.get("content")
        vlm_desc = e.get("vlm_description")

        if isinstance(content, list):
            content_str = " | ".join(f"[{i}]{t}" for i, t in enumerate(content))
            content_str += f"  ※ MUST return smartart_node_index (0 to {len(content)-1})"
        elif content:
            content_str = content[:60]
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
# 組み込みテストケース
# （自前スライドJSONがない場合のデモ用。実際のスライドに合わせて増やすこと）
# ============================================================

BUILTIN_TEST_CASES = [
    # --- テキストマッチング系（比較的簡単） ---
    {
        "id": "TC01",
        "category": "text_direct",
        "description": "直接的なキーワード一致",
        "asr_text": "キーワードマッチングの限界について説明します",
        "expected_element_id": "s1_e4",
        "expected_none": False,
        "elements": [
            {"id": "s1_e1", "type": "text", "shape_name": "Title 1",
             "content": "関連研究：富士通の自動ポインティングシステム"},
            {"id": "s1_e2", "type": "text", "shape_name": "Content",
             "content": "説明音声から資料中の該当箇所をリアルタイムに推定する技術"},
            {"id": "s1_e3", "type": "text", "shape_name": "Content",
             "content": "音声認識辞書を自動生成することで高精度を実現"},
            {"id": "s1_e4", "type": "text", "shape_name": "Content",
             "content": "主にテキストのキーワードマッチングに依存\n図表内の特定の要素や、抽象的な表現には対応困難"},
        ],
    },
    {
        "id": "TC02",
        "category": "text_semantic",
        "description": "意味的理解が必要（キーワード一致なし）",
        "asr_text": "従来手法では図の中身を理解することができませんでした",
        "expected_element_id": "s1_e4",
        "expected_none": False,
        "elements": [
            {"id": "s1_e1", "type": "text", "shape_name": "Title 1",
             "content": "関連研究：富士通の自動ポインティングシステム"},
            {"id": "s1_e2", "type": "text", "shape_name": "Content",
             "content": "説明音声から資料中の該当箇所をリアルタイムに推定する技術"},
            {"id": "s1_e3", "type": "text", "shape_name": "Content",
             "content": "音声認識辞書を自動生成することで高精度を実現"},
            {"id": "s1_e4", "type": "text", "shape_name": "Content",
             "content": "主にテキストのキーワードマッチングに依存\n図表内の特定の要素や、抽象的な表現には対応困難"},
        ],
    },
    # --- SmartArt系 ---
    {
        "id": "TC03",
        "category": "smartart_node",
        "description": "SmartArtの特定ノードを指す",
        "asr_text": "音声認識には faster-whisper を使用しています",
        "expected_element_id": "s2_e2",
        "expected_smartart_node_index": 0,  # "音声認識 (ASR)" ノード
        "expected_none": False,
        "elements": [
            {"id": "s2_e1", "type": "text", "shape_name": "Title 1",
             "content": "システム構成"},
            {"id": "s2_e2", "type": "smartart", "shape_name": "SmartArt 1",
             "content": ["音声認識 (ASR)", "faster-whisperでリアルタイム変換",
                         "意味理解 (LLM)", "Llama3で発話意図を推論",
                         "ポインティング制御", "WebSocketで座標をフロントへ送信"],
             "smartart_node_count": 6},
        ],
    },
    {
        "id": "TC04",
        "category": "smartart_node",
        "description": "SmartArtの別ノードを指す（意味理解）",
        "asr_text": "ポインターはブラウザ上でSVGを使って描画しています",
        "expected_element_id": "s2_e2",
        "expected_smartart_node_index": 4,  # "ポインティング制御" ノード
        "expected_none": False,
        "elements": [
            {"id": "s2_e1", "type": "text", "shape_name": "Title 1",
             "content": "システム構成"},
            {"id": "s2_e2", "type": "smartart", "shape_name": "SmartArt 1",
             "content": ["音声認識 (ASR)", "faster-whisperでリアルタイム変換",
                         "意味理解 (LLM)", "Llama3で発話意図を推論",
                         "ポインティング制御", "WebSocketで座標をフロントへ送信"],
             "smartart_node_count": 6},
        ],
    },
    # --- フィラー・無関係発話（none が正解） ---
    {
        "id": "TC05",
        "category": "none_filler",
        "description": "フィラー発話はポインティング不要",
        "asr_text": "えーと、次のスライドに移ります",
        "expected_element_id": None,
        "expected_none": True,
        "elements": [
            {"id": "s3_e1", "type": "text", "shape_name": "Title 1",
             "content": "実験結果"},
            {"id": "s3_e2", "type": "text", "shape_name": "Content",
             "content": "ポインティング精度: 84%\n平均遅延: 1.8秒"},
        ],
    },
    # --- 指示語（意味的解決が必要） ---
    {
        "id": "TC06",
        "category": "text_pronoun",
        "description": "指示語「ここ」の解消",
        "asr_text": "ここの遅延がボトルネックになっています",
        "expected_element_id": "s3_e3",
        "expected_none": False,
        "elements": [
            {"id": "s3_e1", "type": "text", "shape_name": "Title 1",
             "content": "遅延計測結果"},
            {"id": "s3_e2", "type": "text", "shape_name": "Content",
             "content": "ASR処理時間: 平均 0.85秒"},
            {"id": "s3_e3", "type": "text", "shape_name": "Content",
             "content": "LLM推論時間: 平均 0.95秒 ← ボトルネック"},
            {"id": "s3_e4", "type": "text", "shape_name": "Content",
             "content": "合計遅延: 平均 1.80秒"},
        ],
    },
    # --- 抽象的・文脈依存 ---
    {
        "id": "TC07",
        "category": "text_abstract",
        "description": "抽象的な発話（富士通との差別化）",
        "asr_text": "本研究の新規性は、単語の一致ではなく文脈から意図を読み取る点にあります",
        "expected_element_id": "s4_e3",
        "expected_none": False,
        "elements": [
            {"id": "s4_e1", "type": "text", "shape_name": "Title 1",
             "content": "本研究の位置づけ"},
            {"id": "s4_e2", "type": "text", "shape_name": "Content",
             "content": "富士通 (2015): テキストキーワードマッチング"},
            {"id": "s4_e3", "type": "text", "shape_name": "Content",
             "content": "本研究: LLMによる意味的理解・文脈推論"},
            {"id": "s4_e4", "type": "text", "shape_name": "Content",
             "content": "VLMによる図表内部構造の解析"},
        ],
    },
    # --- 数値・グラフ言及 ---
    {
        "id": "TC08",
        "category": "text_numeric",
        "description": "具体的な数値への言及",
        "asr_text": "精度は84パーセントを達成しました",
        "expected_element_id": "s5_e2",
        "expected_none": False,
        "elements": [
            {"id": "s5_e1", "type": "text", "shape_name": "Title 1",
             "content": "評価結果"},
            {"id": "s5_e2", "type": "text", "shape_name": "Content",
             "content": "ポインティング精度（Accuracy@1）: 84.2%"},
            {"id": "s5_e3", "type": "text", "shape_name": "Content",
             "content": "平均遅延（End-to-End）: 1.8秒"},
            {"id": "s5_e4", "type": "image", "shape_name": "Chart 1",
             "vlm_description": "Bar chart showing accuracy comparison between baseline and proposed method"},
        ],
    },
]


# ============================================================
# テストケース自動生成（スライドJSONから）
# ============================================================

def generate_test_cases_from_json(slide_json_path: str, slide_num: int) -> list[dict]:
    """
    スライドJSONからインタラクティブにテストケースを生成する。
    コマンドラインで発話テキストと正解要素IDを入力させる。
    """
    with open(slide_json_path, encoding="utf-8") as f:
        data = json.load(f)

    elements = []
    for s in data["slides"]:
        if s["slide_number"] == slide_num:
            elements = s["elements"]
            break

    if not elements:
        print(f"[ERROR] スライド {slide_num} の要素が見つかりません")
        return []

    print(f"\n=== スライド {slide_num} の要素一覧 ===")
    for e in elements:
        content = e.get("content")
        if isinstance(content, list):
            preview = " / ".join(content)[:60]
        elif content:
            preview = content[:60]
        else:
            preview = f"[image] {e.get('vlm_description', '')[:40]}"
        print(f"  {e['id']}: [{e['type']}] {preview}")

    cases = []
    print("\n発話テキストと正解要素IDを入力してください（空行で終了）")
    i = 1
    while True:
        print(f"\n--- テストケース {i} ---")
        asr = input("発話テキスト（空行で終了）> ").strip()
        if not asr:
            break
        eid = input("正解 element_id（ポインティング不要なら 'none'）> ").strip()
        expected_none = (eid.lower() == "none")
        cases.append({
            "id": f"TC_custom_{i:02d}",
            "category": "custom",
            "description": f"カスタムケース {i}",
            "asr_text": asr,
            "expected_element_id": None if expected_none else eid,
            "expected_none": expected_none,
            "elements": elements,
        })
        i += 1

    return cases


# ============================================================
# 推論実行
# ============================================================

def run_inference(model: str, asr_text: str, elements: list[dict],
                  verbose: bool = False) -> tuple[dict, float]:
    """
    1回の推論を実行してパース済み結果と所要時間を返す。
    """
    if _TRANSITION_PATTERN.search(asr_text):
        return {"element_id": None, "confidence": "none",
                "reason": "Filtered by transition/filler pattern",
                "smartart_node_index": None}, 0.0
    # タイトル要素を除外（semantic_pointer.py と同じ処理）
    pointable = [e for e in elements
                 if not (e["type"] == "text" and
                         ("タイトル" in e.get("shape_name", "") or
                          "Title" in e.get("shape_name", "")))]

    user_prompt = build_user_prompt(asr_text, pointable)

    t0 = time.time()
    response = _ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        options={"num_predict": 256, "temperature": 0.1},
    )
    elapsed = time.time() - t0

    raw = response["message"]["content"].strip()
    if verbose:
        print(f"    [RAW] {raw[:120]}")

    # JSONパース
    clean = raw
    if "```" in clean:
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    # 閉じ括弧補完
    if clean and not clean.endswith("}"):
        last_comma = clean.rfind(",")
        last_quote = clean.rfind('"')
        if last_quote > last_comma:
            clean = clean[:last_quote + 1] + "\n}"
        elif last_comma > 0:
            clean = clean[:last_comma] + "\n}"
        else:
            clean = clean + "\n}"

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = {
            "element_id": None,
            "confidence": "none",
            "reason": f"[PARSE ERROR] {raw[:80]}",
            "smartart_node_index": None,
        }

    return parsed, elapsed


# ============================================================
# 採点
# ============================================================

def score_result(case: dict, result: dict) -> dict:
    """
    テストケースと推論結果を比較して採点する。

    Returns
    -------
    dict
        correct_element: 要素IDが正しいか (bool)
        correct_node: SmartArtノードインデックスが正しいか (bool | None)
        correct_none: none判定が正しいか (bool)
        overall: 総合的に正解か (bool)
    """
    expected_none  = case.get("expected_none", False)
    expected_eid   = case.get("expected_element_id")
    expected_node  = case.get("expected_smartart_node_index")  # Noneなら評価しない

    pred_conf = result.get("confidence", "none")
    pred_eid  = result.get("element_id")
    pred_node = result.get("smartart_node_index")

    # none判定の正誤
    pred_is_none    = (pred_conf == "none" or pred_eid is None)
    correct_none    = (pred_is_none == expected_none)

    # 要素IDの正誤
    correct_element = (pred_eid == expected_eid) if not expected_none else pred_is_none

    # SmartArtノードインデックスの正誤（ペア構造のためグループ単位で判定）
    correct_node = None
    if expected_node is not None and not expected_none:
        if pred_node is not None:
            # ペア構造（偶数/奇数が同グループ）を考慮
            correct_node = (pred_node // 2 == expected_node // 2)
        else:
            correct_node = False

    # 総合正誤: 要素IDが正しく、ノード指定がある場合はそれも正しい
    if expected_none:
        overall = pred_is_none
    elif expected_node is not None:
        overall = correct_element and (correct_node is True)
    else:
        overall = correct_element

    return {
        "correct_element": correct_element,
        "correct_node":    correct_node,
        "correct_none":    correct_none,
        "overall":         overall,
    }


# ============================================================
# メイン比較ループ
# ============================================================

def run_comparison(
    models: list[str],
    test_cases: list[dict],
    n_runs: int = 3,
    verbose: bool = False,
    output_dir: str = ".",
) -> dict:
    """
    全モデル × 全テストケース × n_runs 回の比較実験を実行する。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path  = os.path.join(output_dir, f"model_comparison_result_{timestamp}.csv")
    summary_path = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.csv")

    # 詳細CSV ヘッダー
    detail_rows = []
    detail_fields = [
        "model", "test_id", "category", "description", "run",
        "asr_text", "expected_eid", "pred_eid", "pred_confidence",
        "pred_node", "elapsed_sec",
        "correct_element", "correct_node", "correct_none", "overall",
        "reason",
    ]

    all_results = {m: [] for m in models}

    for model in models:
        print(f"\n{'='*60}")
        print(f"  モデル: {model}")
        print(f"{'='*60}")

        # モデルが利用可能か事前確認
        try:
            _ollama.chat(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                options={"num_predict": 5},
            )
        except Exception as e:
            print(f"  [SKIP] モデル '{model}' が使用できません: {e}")
            continue

        for case in test_cases:
            tc_id   = case["id"]
            cat     = case["category"]
            desc    = case["description"]
            asr     = case["asr_text"]
            elements = case["elements"]

            print(f"\n  [{tc_id}] {desc}")
            print(f"    ASR: {asr}")

            run_times   = []
            run_results = []

            for run in range(1, n_runs + 1):
                try:
                    result, elapsed = run_inference(model, asr, elements, verbose=verbose)
                    run_times.append(elapsed)
                    run_results.append(result)

                    scores = score_result(case, result)
                    mark   = "✅" if scores["overall"] else "❌"

                    print(f"    Run{run}: {elapsed:.2f}s  "
                          f"id={result.get('element_id')}  "
                          f"conf={result.get('confidence')}  "
                          f"node={result.get('smartart_node_index')}  {mark}")

                    detail_rows.append({
                        "model":           model,
                        "test_id":         tc_id,
                        "category":        cat,
                        "description":     desc,
                        "run":             run,
                        "asr_text":        asr,
                        "expected_eid":    case.get("expected_element_id", ""),
                        "pred_eid":        result.get("element_id", ""),
                        "pred_confidence": result.get("confidence", ""),
                        "pred_node":       result.get("smartart_node_index", ""),
                        "elapsed_sec":     round(elapsed, 3),
                        "correct_element": scores["correct_element"],
                        "correct_node":    scores["correct_node"],
                        "correct_none":    scores["correct_none"],
                        "overall":         scores["overall"],
                        "reason":          result.get("reason", "")[:80],
                    })

                except Exception as e:
                    print(f"    Run{run}: エラー - {e}")
                    run_times.append(None)
                    run_results.append(None)

            # このケースのサマリー
            valid_times  = [t for t in run_times if t is not None]
            valid_results = [r for r in run_results if r is not None]
            valid_scores = [score_result(case, r) for r in valid_results]

            avg_time      = sum(valid_times) / len(valid_times) if valid_times else None
            overall_rate  = sum(s["overall"] for s in valid_scores) / len(valid_scores) if valid_scores else 0

            all_results[model].append({
                "test_id":      tc_id,
                "category":     cat,
                "avg_time":     avg_time,
                "overall_rate": overall_rate,
                "scores":       valid_scores,
            })

    # ============================================================
    # サマリー集計
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  比較結果サマリー")
    print(f"{'='*70}")

    summary_rows = []
    for model, results in all_results.items():
        if not results:
            continue

        all_times   = [r["avg_time"] for r in results if r["avg_time"] is not None]
        all_overall = [r["overall_rate"] for r in results]

        avg_time_total = sum(all_times) / len(all_times) if all_times else 0
        accuracy       = sum(all_overall) / len(all_overall) * 100 if all_overall else 0

        # カテゴリ別精度
        cats = {}
        for r in results:
            c = r["category"]
            if c not in cats:
                cats[c] = []
            cats[c].append(r["overall_rate"])
        cat_acc = {c: sum(v) / len(v) * 100 for c, v in cats.items()}

        print(f"\n  [{model}]")
        print(f"    総合精度    : {accuracy:.1f}% ({sum(r['overall_rate']==1.0 for r in results)}/{len(results)} ケース全正解)")
        print(f"    平均速度    : {avg_time_total:.2f}秒/推論")
        for cat, acc in sorted(cat_acc.items()):
            print(f"    精度 [{cat:<20}]: {acc:.1f}%")

        summary_rows.append({
            "model":        model,
            "accuracy_pct": round(accuracy, 1),
            "avg_time_sec": round(avg_time_total, 3),
            **{f"acc_{c}": round(v, 1) for c, v in cat_acc.items()},
        })

    # ============================================================
    # CSV保存
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)

    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        w.writerows(detail_rows)
    print(f"\n  詳細CSV保存: {detail_path}")

    if summary_rows:
        summary_fields = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"  サマリーCSV保存: {summary_path}")

    return {"detail": detail_rows, "summary": summary_rows}


# ============================================================
# CLIエントリポイント
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLMモデル比較テスト（精度・速度）"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["llama3", "gemma4:12b"],
        help="比較するOllamaモデル名リスト（デフォルト: llama3 gemma4:12b）"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="各ケースの繰り返し回数（デフォルト: 3）"
    )
    parser.add_argument(
        "--slide-json", default=None,
        help="自前スライドJSONからテストケースをインタラクティブ生成"
    )
    parser.add_argument(
        "--slide-num", type=int, default=1,
        help="テストケース生成に使うスライド番号"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="CSV出力ディレクトリ（デフォルト: カレントディレクトリ）"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="LLMの生レスポンスを表示"
    )
    args = parser.parse_args()

    # テストケースの準備
    if args.slide_json:
        if not os.path.exists(args.slide_json):
            sys.exit(f"[ERROR] JSONが見つかりません: {args.slide_json}")
        test_cases = generate_test_cases_from_json(args.slide_json, args.slide_num)
        if not test_cases:
            print("[INFO] テストケースが0件のため、組み込みケースを使用します")
            test_cases = BUILTIN_TEST_CASES
    else:
        test_cases = BUILTIN_TEST_CASES

    print(f"\n比較モデル : {args.models}")
    print(f"テストケース: {len(test_cases)}件")
    print(f"繰り返し   : {args.runs}回/ケース")
    print(f"総推論回数 : {len(args.models) * len(test_cases) * args.runs}回")

    run_comparison(
        models=args.models,
        test_cases=test_cases,
        n_runs=args.runs,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
