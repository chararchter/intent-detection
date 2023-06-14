import requests
from transformers import pipeline

from secret_api_token import API_TOKEN

text_to_translate = "kad ir nākamais vilciens"

# use local model
# git clone https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-lv-en

pipe = pipeline("translation", model=".\opus-mt-tc-big-lv-en")
translated_text = pipe(text_to_translate)

print(translated_text)


# use API
def query(payload, model_id, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


model_id = "Helsinki-NLP/opus-mt-tc-big-lv-en"

translated_text = query(text_to_translate, model_id, API_TOKEN)
print(translated_text)
