import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pyaudio
import wave
import time
from faster_whisper import WhisperModel
import collections
import numpy as np

# --- 設定 --- #
# faster-whisper モデル設定
# VRAM 4GBなのでsmallまたはbaseを推奨
MODEL_SIZE = "small"
DEVICE = "cuda"  # GPUを使用
COMPUTE_TYPE = "float16" # VRAM節約のためfloat16を推奨

# PyAudio 設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisperモデルの推奨サンプリングレート
CHUNK = 1024  # 1回の読み込みで処理するフレーム数
RECORD_SECONDS = 3 # 3秒ごとに文字起こしを実行

# --- Whisperモデルのロード --- #
print(f"Loading faster-whisper {MODEL_SIZE} model on {DEVICE} with {COMPUTE_TYPE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded.")

# --- PyAudioの初期化 --- #
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording... Press Ctrl+C to stop.")

frames = collections.deque()
last_transcription_time = time.time()

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # 指定秒数ごとに文字起こしを実行
        if time.time() - last_transcription_time >= RECORD_SECONDS:
            # 過去の音声データを結合
            audio_data = b''.join(list(frames))
            # NumPy配列に変換 (Whisperモデルの入力形式に合わせる)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # 文字起こし
            segments, info = model.transcribe(
                audio_np, 
                beam_size=5, 
                language="ja", 
                vad_filter=True,  # 声がない区間を無視する
                vad_parameters=dict(min_silence_duration_ms=500) # 0.5秒以上の無音で区切る
                )
            
            # transcribed_text = "".join([segment.text for segment in segments])
            # if transcribed_text.strip(): # 空文字列でなければ表示
            #     print(f"[ASR] {transcribed_text}")
            for segment in segments:
                text = segment.text.strip()
                # 信頼度（avg_logprob）が -1.0 より大きく、かつ2文字以上の時だけ表示
                # ※ -1.0 は「かなり自信がある」状態です。もし何も出なくなったら -1.5 くらいに緩めてください。
                if segment.avg_logprob > -1.0:
                    is_kanji = any('\u4e00' <= char <= '\u9faf' for char in text)
                    if len(text) > 1 or (len(text) == 1 and is_kanji):
                        print(f"[ASR] {text} (信頼度: {segment.avg_logprob:.2f})")
            # 処理した音声データはクリアし、次のセグメントに備える
            frames.clear()
            last_transcription_time = time.time()

except KeyboardInterrupt:
    print("Stopping recording.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Audio stream closed.")
