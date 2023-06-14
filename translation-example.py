import json

import requests
from transformers import pipeline

from secret_api_token import API_TOKEN

# use local model
# git clone https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-lv-en
text_to_translate = "kad ir nākamais vilciens"

pipe = pipeline("translation", model=".\opus-mt-tc-big-lv-en")
translated_text = pipe(text_to_translate)

print(translated_text)

# use API
API_URL_LV = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-lv-en"

headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload: str, api_url: str) -> dict:
    data = json.dumps(payload)
    response = requests.request("POST", api_url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


translated_text = query(text_to_translate, API_URL_LV)
print(translated_text)
