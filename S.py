import json
with open('test_new_prompt.json', encoding='utf-8') as f:
    data = json.load(f)
for s in data['slides']:
    if s['slide_number'] == 4:
        for e in s['elements']:
            if e['id'] == 's3_e4':
                print(e.get('vlm_description', ''))
