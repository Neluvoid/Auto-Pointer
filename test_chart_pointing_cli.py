"""
chart_pointing 単体テストスクリプト v4
変更点:
- ケース10: 「東京の売上を見てください」の期待値を none に修正
  （系列全体への言及はポインティング不要、発話が曖昧すぎる）
- ケース11: #コメント対応後の再テスト
- ケース13を追加: より明確な特定バー指定
"""
import sys, os, json, argparse, base64

sys.path.insert(0, r"C:\Users\admin\GitHub\Auto-Pointer")
from semantic_pointer import get_chart_pointing, _make_bar_position_hint, _make_trend_arrow_hint

VLM_LINE_TEMP = (
    "This is a line chart titled '気温の変化' (Temperature change). "
    "The x-axis shows time from 7時 (7 AM) to 19時 (7 PM). "
    "The y-axis shows temperature in Celsius, ranging from 0 to 25. "
    "The line starts at 13.7℃ at 7時, rises to a peak of 23.1℃ at 14時, "
    "then decreases back to 13.8℃ at 19時. Shape: inverted U-curve."
)
VLM_BAR_FUEL = (
    "This image is a vertical bar chart illustrating the total fueling cost for three "
    "vehicles: Gasoline Car, Hybrid Car, and Electric Vehicle (EV). "
    "Order of bars from left to right:\n"
    "1. Gasoline Car (highest bar, ~$3000)\n"
    "2. Hybrid Car (medium bar, ~$1800)\n"
    "3. Electric Vehicle (~$900)\n"
    "The bars decrease in height from left to right."
)
VLM_BAR_SALES = (
    "This is a grouped bar chart titled '月別売上' (Monthly Sales). "
    "The x-axis shows three months: 1月 (January), 2月 (February), 3月 (March). "
    "Each month has 3 bars: 東京 (Tokyo, blue, ~13-15M yen), "
    "大阪 (Osaka, green, ~12-18M yen), 名古屋 (Nagoya, light blue, ~7-8M yen). "
    "In January: Osaka (18M) > Tokyo (13M) > Nagoya (8M). "
    "In February and March: Tokyo (15M) > Osaka (12M) > Nagoya (7M). "
    "Y-axis: 0 to 20,000,000 yen."
)
BBOX = [0.05, 0.05, 0.90, 0.90]

def load_b64(path):
    if not path or not os.path.exists(path): return None
    try:
        from PIL import Image; import io
        img = Image.open(path).convert("RGB")
        if max(img.size) > 1024:
            r = 1024/max(img.size); img = img.resize((int(img.size[0]*r), int(img.size[1]*r)), Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[WARN] {e}"); return None

def run_tests(line_img=None, bar_img=None, gbar_img=None, model="ministral-3:8b", verbose=False):
    lb = load_b64(line_img); bb = load_b64(bar_img); gb = load_b64(gbar_img or bar_img)
    print(f"折れ線: {'あり' if lb else 'なし'}  棒グラフ: {'あり' if bb else 'なし'}  グループ棒: {'あり' if gb else 'なし'}")
    print(f"モデル: {model}\n")

    TESTS = [
        # 折れ線
        ("右肩上がりになっています",    VLM_LINE_TEMP, lb, "arrow",  "折れ線・上昇"),
        ("午後から下がっています",       VLM_LINE_TEMP, lb, "arrow",  "折れ線・後半下降"),
        ("14時がピークです",             VLM_LINE_TEMP, lb, "circle", "折れ線・特定点"),
        ("このグラフについて説明します", VLM_LINE_TEMP, lb, "none",   "折れ線・全体"),
        # 単純棒グラフ
        ("だんだん左から右に下がっています", VLM_BAR_FUEL, bb, "arrow",  "棒・減少トレンド"),
        ("電気自動車が最も安いです",     VLM_BAR_FUEL, bb, "circle", "棒・特定バー(最小)"),
        ("ガソリン車の費用が最大です",   VLM_BAR_FUEL, bb, "circle", "棒・特定バー(最大)"),
        ("このグラフについて説明します", VLM_BAR_FUEL, bb, "none",   "棒・全体"),
        # グループ化棒グラフ
        ("1月は大阪が最も売上が高いです",VLM_BAR_SALES, gb, "circle", "グループ棒・特定バー"),
        ("東京の売上を見てください",     VLM_BAR_SALES, gb, "none",   "グループ棒・系列言及→none"),
        ("名古屋は3ヶ月とも低いですね",  VLM_BAR_SALES, gb, "circle", "グループ棒・系列全体"),
        ("このグラフについて説明します", VLM_BAR_SALES, gb, "none",   "グループ棒・全体"),
        ("2月の名古屋はいくらですか",    VLM_BAR_SALES, gb, "circle", "グループ棒・特定バー指定"),
    ]

    correct = 0
    for i, (asr, vlm, img_b64, expected, label) in enumerate(TESTS, 1):
        print(f"[{i:2d}] {label}")
        print(f"     発話: {asr}  期待: {expected}")
        hint = "BAR" if _make_bar_position_hint(vlm) else ("TREND" if _make_trend_arrow_hint(vlm) else "なし")
        result = get_chart_pointing(asr, vlm, BBOX, img_b64, model, verbose)
        if result is None:
            print(f"     結果: None ❌\n"); continue
        dt = result.get("draw_type","N/A")
        ok = "✅" if dt == expected else "❌"
        print(f"     結果: {dt} {ok}  hint={hint}")
        if result.get("abs_bbox_ratio"): print(f"     bbox: {[round(v,3) for v in result['abs_bbox_ratio']]}")
        if result.get("abs_arrow_start"): print(f"     arrow: {result['abs_arrow_start']} -> {result['abs_arrow_end']}")
        print(f"     reason: {result.get('reason','')[:65]}\n")
        if dt == expected: correct += 1

    print("=" * 65)
    print(f"精度: {correct}/{len(TESTS)} ({correct/len(TESTS)*100:.0f}%)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--line-image", default=None)
    p.add_argument("--bar-image",  default=None)
    p.add_argument("--grouped-image", default=None)
    p.add_argument("--image",      default=None)
    p.add_argument("--model",      default="ministral-3:8b")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()
    run_tests(args.line_image or args.image, args.bar_image, args.grouped_image, args.model, args.verbose)
