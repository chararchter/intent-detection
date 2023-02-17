import json
from typing import List

import requests

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


def get_data(path: str) -> List[str]:
    """ Read text file in the specified path and append each line without \n as an element to an array.
    Encoding is specified to correctly read files in Russian.

    :param path: path e.g. "NLU-datasets\chatbot\chatbot_train_ans.txt"
    :return: array e.g. ['FindConnection', 'FindConnection', ..., 'FindConnection']
    """
    print(path)
    with open(path, encoding='utf-8') as f:
        array = []
        for line in list(f):
            array.append(line.split('\n')[0])
        return array


def read_source_text(dataset_type: str, source_language: str = None, labels: bool = True) -> List[str]:
    """ Wrapper for get_data that provides file path.
    Prompts in all languages are in the same order, therefore they use the same label files. So please be careful
    to use the correct argument for labels, as label=True returns labels regardless of specified source_language

    Usage examples:
    prompts: read_source_text("test", "et", False)

    labels: read_source_text("test")

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et", "lt"
    :param labels: does the file being read contain labels
    :return: array of file contents for specified file
    """
    if labels:
        return get_data(f"NLU-datasets\chatbot\chatbot_{dataset_type}_ans.txt")
    else:
        return get_data(f"NLU-datasets\chatbot\{source_language}\chatbot_{dataset_type}_q.txt")


def translate_to_file(source_language: str, dataset_type: str, dataset: List[str], api_url: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset_type: "test" or "train"
    :param translated_text: array of arrays with one dictionary where key='translation_text' and value is the translated text
    e.g. [[{'translation_text': "Taxi's waiting."}]]
    """
    with open(f"{source_language}_{dataset_type}.txt", "w", encoding="utf-8") as f:
        for line in dataset:
            output = query(line, api_url)
            if "error" in output:
                print(output)
                f.write("error, og line:" + line + "\n")
            else:
                f.write(output[0]["translation_text"] + "\n")


def translate(dataset: List[str], api_url: str) -> List[str]:
    """ Iterate through training set and translate each line
    """
    array = []
    for line in dataset:
        output = query(line, api_url)

        if "error" in output:
            raise ValueError("Model is currently loading or Service Unavailable, try again")
        array.append(output)
    return array


lv_test = read_source_text("test", "lv", False)
ru_test = read_source_text("test", "ru", False)
et_test = read_source_text("test", "et", False)
lt_test = read_source_text("test", "lt", False)

lv_train = read_source_text("train", "lv", False)
ru_train = read_source_text("train", "ru", False)
et_train = read_source_text("train", "et", False)
lt_train = read_source_text("train", "lt", False)


translate_to_file(source_language="lv", dataset_type="test", dataset=lv_test, api_url=API_URL_LV)
