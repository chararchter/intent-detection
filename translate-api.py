import json
from typing import List

import requests

from model import get_source_text
from secret_api_token import API_TOKEN

API_URL_LV = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-lv-en"
API_URL_RU = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ru-en"
API_URL_ET = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-et-en"
API_URL_LT = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-lt-en"

headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload: str, api_url: str) -> dict:
    data = json.dumps(payload)
    response = requests.request("POST", api_url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def translate_to_file(dataset_type: str, source_language: str, dataset_name: str, dataset: List[str], api_url: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset_name: "chatbot", "askubuntu" or "webapps"
    :param dataset: dataset of source_language and type e.g. "lv_train"
    :param api_url: API endpoint
    """
    with open(f"{dataset_name}_{source_language}_{dataset_type}.txt", "w", encoding="utf-8") as f:
        for line in dataset:
            output = query(line, api_url)
            if "error" in output:
                print(output)
                f.write("error, original line:" + line + "\n")
            else:
                f.write(output[0]["translation_text"] + "\n")


def translate(dataset: List[str], api_url: str) -> List[str]:
    """ Iterate through training set and translate each line. Not using this, because it is easier to
    translate a few missing lines than terminate on the first error
    """
    array = []
    for line in dataset:
        output = query(line, api_url)
        if "error" in output:
            raise ValueError("Model is currently loading or Service Unavailable, try again")
        array.append(output)
    return array


def read_and_translate(source_language: str, dataset_name: str, dataset_type: str, api_url: str):
    dataset = get_source_text(dataset_type, dataset_name, source_language)
    print(dataset)
    translate_to_file(
        source_language=source_language,
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        dataset=dataset,
        api_url=api_url
    )


api = {
    "lv": API_URL_LV,
    "ru": API_URL_RU,
    "et": API_URL_ET,
    "lt": API_URL_LT,
}

for language, api_url in api.items():
    for dataset_name in ["chatbot", "webapps", "askubuntu"]:
        for dataset_type in ["test", "train"]:
            read_and_translate(
                source_language=language, dataset_name=dataset_name, dataset_type=dataset_type, api_url=api_url
            )
