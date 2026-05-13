
import json, base64, io
from PIL import Image

with open('server_output/AI自動ポインティングシステムの発案_slide_data.json', encoding='utf-8') as f:
    data = json.load(f)

for s in data['slides']:
    if s['slide_number'] == 4:
        for e in s['elements']:
            if e['id'] == 's3_e4':
                bbox_ratio = e['bbox_ratio']
                print('bbox_ratio:', bbox_ratio)
                img_path = s.get('image_path')
                print('image_path:', img_path)
                
                img = Image.open(img_path).convert('RGB')
                W, H = img.size
                print('slide size:', W, H)
                rx, ry, rw, rh = bbox_ratio
                crop = img.crop((int(rx*W), int(ry*H), int((rx+rw)*W), int((ry+rh)*H)))
                print('crop size:', crop.size)
                crop.save('debug_crop.png')
                print('saved: debug_crop.png')
