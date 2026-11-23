import ollama, base64, io
from PIL import Image

img = Image.open('vlm_comparison_output/slide4_s3_e4.png').convert('RGB')
buf = io.BytesIO()
img.save(buf, format='PNG')
img_b64 = base64.b64encode(buf.getvalue()).decode()

vlm_desc = '''Bar Chart. Order of Bars (Left to Right):
1. Ford F150 (Regular Cab)
2. Toyota RAV4
3. Honda Civic
Values: Ford F150: \$130.96, Toyota RAV4: \$82.56, Honda Civic: \$70.55.
Trend: decreasing left to right.'''

prompt = f'''Speaker's utterance (Japanese): \"トヨタRAV4\"

Chart description (read carefully for bar order left to right):
{vlm_desc}

IMPORTANT: Use the bar order from the description above to identify which bar the speaker is referring to.
Then provide the bounding box for that specific bar.

Reply ONLY with JSON:
{{\"draw_type\": \"circle\", \"bbox_ratio\": [x_left, y_top, width, height], \"arrow_start_ratio\": null, \"arrow_end_ratio\": null, \"reason\": \"which bar (left/middle/right) was selected and why\"}}'''

r = ollama.chat(
    model='ministral-3:8b',
    messages=[{'role':'user','content':prompt,'images':[img_b64]}]
)
print(r['message']['content'][:300])
