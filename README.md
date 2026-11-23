
# Automatic Slide Pointing System Based on Voice Semantic Understanding
## Prototype Development Document

**Author:** Yo Shiraki (M1)
**Last Updated:** May 2026
**Repository:** Auto-Pointer (GitHub)

---

## 1. System Overview

### 1.1 Purpose

A system that uses AI to understand the meaning of the presenter's voice in real time during a presentation and automatically points to the appropriate location on the slide (text, charts, specific SmartArt nodes, etc.). The aim is to reduce the burden of pointer operation on the presenter and guide the audience's gaze more naturally and effectively.

### 1.2 Features

- **Semantic Understanding-Based Pointing:** Instead of keyword matching, LLM (Llama3) understands the intent and context of the utterance to infer the target element.
- **Automatic Slide Structure Analysis:** Extracts coordinates of text, shapes, and SmartArt from PPTX files and uses them as the basis data for pointing.
- **Network Operation:** Supports a configuration with a server PC equipped with a high-performance GPU and a laptop PC with a built-in microphone.
- **Browser-Based UI:** Slide display, pointer drawing, and accuracy evaluation are all completed within a web browser.

---

## 2. System Architecture

### 2.1 Overall Configuration

```
Laptop PC (Presenter) Server PC (AI Processing)
┌─────────────────────┐ ┌──────────────────────────────┐
│ Chrome (HTTPS Tunnel) │ ←WebSocket→ │ FastAPI + Socket.IO │
│ ・Slide Display │ │ ・Whisper (Voice Recognition) │
│ ・Pointer Drawing │ │ ・Llama3 (Semantic Understanding) │
│ ・Microphone Input │ │ ・slide_parser (PPTX Parsing) │
│ ・Accuracy Evaluation UI │ │ ・gemma3:12b (ASR Prompt) │
└────────────────────────┘ └──────────────────────────────┘
```

### 2.2 Processing Flow

```
1. PPTX Upload

└→ Extract text, shapes, and SmartArt coordinates with slide_parser.py (convert to JSON)

└→ Export slide images with PowerPoint COM

└→ Generate ASR initial_prompt for each slide with gemma3:12b

2. Voice Input (Real-time)

Browser microphone → ScriptProcessorNode → 3-second chunk → base64 encoding

└→ Send to server via Socket.IO audio_chunk event

3. Speech Recognition (ASR)

Whisper medium → Text (avg_logprob filtering + silent skip)

4. Semantic Understanding (LLM)

Llama3 + slide element metadata → Pointing target element ID + bbox_ratio + confidence

5. Pointer Drawing

Socket.IO pointing_result → SVG overlay drawing

(Smooth movement using lerp interpolation, pulse animation after arrival)
```

---

## 3. Implementation File Structure

```
Auto-Pointer/
├── server.py # FastAPI + Socket.IO server (main entry point)
├── semantic_pointer.py # Semantic understanding module (LLM inference)
├── slide_parser.py # PPTX Slide Parsing Module
├── draw_pointer.py # Pointer drawing module (for still image output)
├── mvp_test.py # Offline integration test script
├── asr_demo.py # ASR standalone demo script
├── speed_test.py # LLM speed test script
├── static/
│ └── index.html # Frontend (slide display, pointer UI)
└── logs/ # Server logs (automatically generated)
```

---

## 4. Details of Main Modules

### 4.1 slide_parser.py

Parses PPTX files and outputs element metadata for each slide in JSON format.


**Extracted Information:**
- Text Box: Text content and coordinates (bbox_pt / bbox_ratio)
- SmartArt: Text node list, number of nodes, and coordinates
- Image/Shape: Type and coordinates (VLM content analysis option)
- Table: Cell content and coordinates

**Example Commands:**
```bash
# Basic analysis (without VLM)
python slide_parser.py your_slide.pptx --no-vlm --output slide_data.json

# Generate ASR initial_prompt simultaneously
python slide_parser.py your_slide.pptx --no-vlm --gen-asr-prompts --output slide_data.json

# Export slide images as well (Windows + PowerPoint required)
python slide_parser.py your_slide.pptx --no-vlm --export-images
```

### 4.2 semantic_pointer.py

Receives ASR text and slide element metadata, and uses Llama3 to infer the pointing target.

**Input:**
- `asr_text`: Text obtained by speech recognition (1 sentence)
- `slide_elements`: List of slide elements output by slide_parser.py
**Output (dict):**
```json
{
"element_id": "s4_e1",
"element_type": "smartart",
"content_preview": "Semantic understanding expression / Simple words...",
"bbox_ratio": [0.078, 0.187, 0.854, 0.739],
"smartart_node_index": 0,
"smartart_node_bbox_ratio": [0.078, 0.220, 0.854, 0.245],
"confidence": "high",
"reason": "The speaker mentions semantic Understanding..."
}
```
**SmartArt Support:**
Detects pair structures of even-indexed headings and odd-indexed descriptions, and improves pointing accuracy by dividing bboxes into groups.

### 4.3 server.py

WebSocket server using FastAPI + python-socketio.

**Main Endpoints:**

| Endpoint | Description |
|---|---|
| `GET /` | Frontend HTML Delivery |
| `GET /api/slides` | Slide Metadata List |
| `GET /api/slide/{n}/image` | Slide Image (PNG) |
| `POST /api/upload/pptx` | PPTX Upload & Setup |
| `GET /api/timing/csv` | CSV Download of Latency Measurement Data |
| `GET /api/pptx/status` | Setup Progress |

**Socket.IO Events:**

| Event | Direction | Description |
|---|---|---|
| `audio_chunk` | Client → Server | Audio Chunk (base64 Float32 16kHz) |
| `asr_text` | Client → Server | Manual Text Input |
| `change_slide` | Client → Server | Slide Switch |
| `pointing_result` | Server → Client | Pointing Result |
| `asr_transcript` | Server → Client | Speech Recognition Text (Subtitles) |
| `pptx_progress` | Server → Client | PPTX Setup Progress |
| `asr_prompt_progress` | Server → Client | ASR Prompt Generation Progress |
| `asr_prompt_ready` | Server → Client | ASR Prompt Generation Complete |

### 4.4 index.html (Frontend)

Slide display, pointer drawing, microphone input, and accuracy evaluation UI are all implemented in a single file.

**Key Features:**
- Drag and drop upload of PPTX files
- Waiting overlay during ASR prompt generation (input disabled until generation complete)
- Pointer drawing with SVG overlay (highlight, border, laser)
- Smooth pointer movement with lerp interpolation (dynamic LERP: speed changes according to distance)
- Color coding by confidence level (high=red, medium=orange, low=yellow)
- Pulse animation after arrival
- Focus mode (F key to hide panel, full-width slide display)
- Accuracy evaluation UI (✓ Correct/✗ Incorrect buttons, real-time accuracy % display, CSV export)

---

## 5. Operating Environment

### 5.1 Server PC (AI Processing)

| Item | Specifications |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 8GB |
| OS | Windows 11 |
| Python | 3.10 (Miniconda environment: auto_pointer) |
| Ollama | llama3, gemma3:12b |
| Whisper | faster-whisper medium |

### 5.2 Client PC (Presenter)

| Item | Specifications |
|---|---|
| PC | Notebook PC (RTX 3050 Ti Laptop GPU) |
| Browser | Google Chrome (--unsafely-treat-insecure-origin-as-secure flag enabled) |
| Microphone | Built-in microphone or external microphone |

### 5.3 Network

Access within the same LAN, or via an HTTPS tunnel such as devtunnels.

The browser's microphone API only works in an HTTPS (or localhost) environment, so an HTTPS tunnel is required.

---
## 6. How to Start

### 6.1 Server Startup
```powershell
conda activate auto_pointer
cd C:\path\to\Auto-Pointer
# Standard Startup (Browser ASR Mode)
python server.py --model llama3 --asr-prompt-model gemma3:12b --asr-model medium
# When Using Server-Side Microphone (Local Operation)
python server.py --model llama3 --asr-prompt-model gemma3:12b --asr-model medium --asr
```

### 6.2 Client Access

**On the Same PC (localhost):**
```
http://localhost:8000
```

**Over the Network (Chrome Shortcut):**
```
"C:\Program Files\Google\Chrome\Application\chrome.exe"
--unsafely-treat-insecure-origin-as-secure=http://192.168.x.x:8000
--user-data-dir=C:\tmp\chrome-dev
```
**When using devtunnels:**
```powershell
devtunnel host -p 8000
# → Accessible at https://XXXXXX-8000.jpe1.devtunnels.ms
```

### 6.3 Usage Procedure

1. Access the server URL in your browser.
2. Drag and drop (or click to select) the PPTX file into the drop zone at the top of the panel.
3. Slide analysis and image export will be performed automatically (approximately 10-30 seconds).
4. Start using when the ASR prompt generation completion waiting overlay disappears.
5. Click the microphone button to start browser ASR.
6. Pointing will be performed when you speak.
7. Record the evaluation of each pointing result with "✓ Correct / ✗ Incorrect".
8. After completion, download the evaluation data using "⬇ Accuracy CSV"

---

## 7. Performance Measurement Results

### 7.1 Latency Measurement (RTX 5060 Ti 8GB)

| Item | Measured Value |
|---|---|
| ASR (Whisper medium) | Approx. 0.8-0.9 seconds |
| LLM (Llama3 8B) | Approx. 0.7-1.1 seconds |
| **Total (Speech to Pointing)** | **Approx. 1.6-2.0 seconds** |

### 7.2 Reference: Old Environment (RTX 3050 Ti Laptop GPU)

| Item | Measured Value |
|---|---|
| ASR (Whisper small) | Approx. 0.03 seconds |
| LLM (Llama3 8B) | Approx. 8-9 seconds |
| **Total** | **Approximately 8-9 seconds** |

---
## 8. Known Issues and Limitations

| Problem | Situation | Countermeasures |
|---|---|---|
| Auditory hallucinations in silent chunks | Resolved | Skip chunks with max amplitude < 0.01 |
| SmartArt bbox accuracy | May deviate from actual layout due to equal division estimation | Considering improvement in LibreOffice rendering or PowerPoint COM |
| Pointer reset when switching focus modes | Acceptable | Switching and pointing do not occur simultaneously in actual presentations |
| slide_parser.py print log not integrated into logging | Acceptable | Due to being a function on the mvp_test.py side |
| Pointing delay (approximately 1.6 seconds) | Acceptable | Practical level for demo |

---

## 9. Future Challenges

### 9.1 Accuracy Improvement
- Accuracy Comparison Experiment When Switching LLM to Llama3.1:70b (16GB VRAM Environment)
- Consideration of Switching to Whisper large-v3 (if VRAM capacity becomes available)
- Implementation of logic to resolve demonstrative pronouns ("this," "here")
- Support for elements within figures and tables (trends of bars and lines in graphs, etc.)

### 9.2 Functionality Enhancement
- Selection logic to control excessive pointing (instructor feedback)
- Improvement of continuous pointing through accumulation of speech context
- Deep analysis of internal figure and table structure through integration with VLM (Moondream2)

### 9.3 Evaluation Experiment
- Accuracy evaluation experiment across all slides (utilizing CSV data from the accuracy evaluation UI)
- Comparison of three conditions: manual pointing, keyword matching, and this system
- User study (subjective evaluation of presenter burden and audience comprehension)

---
