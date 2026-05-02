import time
import ollama

prompt_heavy = """Speaker: 'キーワードマッチング'
s1_e1: 説明音声から資料中の該当箇所をリアルタイムに推定する技術（富士通, 2015）
s1_e4: 主にテキストのキーワードマッチングに依存
図表内の特定の要素や、抽象的な表現を用いた説明には対応が困難
Respond JSON: {element_id, confidence, reason, smartart_node_index}"""

prompt_light = """Speaker: 'キーワードマッチング'
s1_e1: 説明音声から資料中の該当箇所をリアルタイムに推定する技術（富士通
s1_e4: 主にテキストのキーワードマッチングに依存
Respond JSON: {element_id, confidence, reason, smartart_node_index}"""

for label, p in [("heavy", prompt_heavy), ("light", prompt_light)]:
    t = time.time()
    r = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": p}],
        options={"temperature": 0.1, "num_predict": 80}
    )
    elapsed = time.time() - t
    content = r["message"]["content"][:80]
    print(f"{label}: {elapsed:.1f}s")
    print(f"  -> {content}")
    print()
