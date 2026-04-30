import ollama
import base64
import os

# --- 設定 --- #
# Llama 3 (テキスト理解用) モデル名
LLM_MODEL = "llama3"
# Moondream2 (画像理解用) モデル名
VLM_MODEL = "moondream"

# --- テキスト理解のデモ (Llama 3) --- #
def demonstrate_llm_text_understanding(prompt):
    print(f"\n--- LLM ({LLM_MODEL}) テキスト理解デモ ---")
    print(f"Prompt: {prompt}")
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
        print("Response:")
        print(response['message']['content'])
    except Exception as e:
        print(f"Error with LLM: {e}")
        print(f"Please ensure '{LLM_MODEL}' model is downloaded and running: 'ollama run {LLM_MODEL}'")

# --- 画像理解のデモ (Moondream2) --- #
def demonstrate_vlm_image_understanding(image_path, prompt):
    print(f"\n--- VLM ({VLM_MODEL}) 画像理解デモ ---")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        response = ollama.chat(model=VLM_MODEL, messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_data]
            }
        ])
        print("Response:")
        print(response['message']['content'])
    except Exception as e:
        print(f"Error with VLM: {e}")
        print(f"Please ensure '{VLM_MODEL}' model is downloaded and running: 'ollama run {VLM_MODEL}'")


if __name__ == "__main__":
    # LLMデモの実行
    #llm_prompt = "AIポインティングシステムとは何ですか？その主要なコンポーネントを教えてください。"
    # demonstrate_llm_text_understanding(llm_prompt)

    # VLMデモの実行
    # ここに以前アップロードしたグラフ画像のパスを指定してください
    # 例: image_file = "/home/ubuntu/upload/1-20220311143337.png"
    # ユーザーが提供した画像パスを使用
    image_file = r"C:\Users\wtaki\GitHub\Auto-Pointer\1-20220311143337.png"
    vlm_prompt = "This graph shows the population by year. Which year has the largest population? "
    demonstrate_vlm_image_understanding(image_file, vlm_prompt)

    print("\n--- デモ完了 ---\n")
    print("これらのデモを実行する前に、Ollamaサーバーが起動しており、")
    print(f"'{LLM_MODEL}' と '{VLM_MODEL}' モデルがダウンロードされていることを確認してください。")
    print(f"ダウンロードコマンド: 'ollama run {LLM_MODEL}' と 'ollama run {VLM_MODEL}'")
    print("また、Pythonライブラリ 'ollama' がインストールされている必要があります: 'pip install ollama'")
