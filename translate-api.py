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


def read_file(path: str) -> List[str]:
    """ Read text file in the specified path and append each line without \n as an element to an array.
    Encoding is specified to correctly read files in Russian.

    :param path: path e.g. "NLU-datasets\chatbot\chatbot_train_ans.txt"
    :return: array e.g. ['FindConnection', 'FindConnection', ..., 'FindConnection']
    """
    # print(path)
    with open(path, encoding='utf-8') as f:
        array = []
        for line in list(f):
            array.append(line.split('\n')[0])
        return array


def get_source_text(dataset_type: str, source_language: str) -> List[str]:
    """ Wrapper for get_data that provides file path.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et", "lt"
    :return: array of file contents for specified file
    """
    return read_file(f"NLU-datasets\chatbot\{source_language}\chatbot_{dataset_type}_q.txt")


def translate_to_file(dataset_type: str, source_language: str, dataset: List[str], api_url: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset: dataset of source_language and type e.g. "lv_train"
    :param api_url: API endpoint
    """
    with open(f"{source_language}_{dataset_type}.txt", "w", encoding="utf-8") as f:
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


lv_test = get_source_text("test", "lv")
ru_test = get_source_text("test", "ru")
et_test = get_source_text("test", "et")
lt_test = get_source_text("test", "lt")

lv_train = get_source_text("train", "lv")
ru_train = get_source_text("train", "ru")
et_train = get_source_text("train", "et")
lt_train = get_source_text("train", "lt")


translate_to_file(source_language="lv", dataset_type="test", dataset=lv_test, api_url=API_URL_LV)
translate_to_file(source_language="lv", dataset_type="train", dataset=lv_train, api_url=API_URL_LV)

translate_to_file(source_language="ru", dataset_type="test", dataset=ru_test, api_url=API_URL_RU)
translate_to_file(source_language="ru", dataset_type="train", dataset=ru_train, api_url=API_URL_RU)

translate_to_file(source_language="et", dataset_type="test", dataset=et_test, api_url=API_URL_ET)
translate_to_file(source_language="et", dataset_type="train", dataset=et_train, api_url=API_URL_ET)

translate_to_file(source_language="lt", dataset_type="test", dataset=lt_test, api_url=API_URL_LT)
translate_to_file(source_language="lt", dataset_type="train", dataset=lt_train, api_url=API_URL_LT)
