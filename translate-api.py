import json
from typing import List

import requests

from model import read_file
from secret_api_token import API_TOKEN

API_URL_LV = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-lv-en"
API_URL_RU = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ru-en"
API_URL_ET = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-et-en"
API_URL_LT = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-lt-en"

CHATBOT = "chatbot"
WEBAPPS = "webapps"
UBUNTU = "askubuntu"

TEST = "test"
TRAIN = "train"

headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload: str, api_url: str) -> dict:
    data = json.dumps(payload)
    response = requests.request("POST", api_url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def get_source_text(dataset_type: str, source_language: str, dataset_name: str) -> List[str]:
    """ Wrapper for get_data that provides file path.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et", "lt"
    :param dataset_name: "chatbot", "askubuntu" or "webapps"
    :return: array of file contents for specified file
    """
    return read_file(f"NLU-datasets\{dataset_name}\{source_language}\{dataset_name}_{dataset_type}_q.txt")


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



lv_test = get_source_text(TEST, "lv", CHATBOT)
ru_test = get_source_text(TEST, "ru", CHATBOT)
et_test = get_source_text(TEST, "et", CHATBOT)
lt_test = get_source_text(TEST, "lt", CHATBOT)

lv_train = get_source_text(TRAIN, "lv", CHATBOT)
ru_train = get_source_text(TRAIN, "ru", CHATBOT)
et_train = get_source_text(TRAIN, "et", CHATBOT)
lt_train = get_source_text(TRAIN, "lt", CHATBOT)

translate_to_file(source_language="lv", dataset_type=TEST, dataset_name=CHATBOT, ataset=lv_test, api_url=API_URL_LV)
translate_to_file(source_language="lv", dataset_type=TRAIN, dataset_name=CHATBOT, dataset=lv_train,
                  api_url=API_URL_LV)

translate_to_file(source_language="ru", dataset_type=TEST, dataset_name=CHATBOT, dataset=ru_test, api_url=API_URL_RU)
translate_to_file(source_language="ru", dataset_type=TRAIN, dataset_name=CHATBOT, dataset=ru_train,
                  api_url=API_URL_RU)

translate_to_file(source_language="et", dataset_type=TEST, dataset_name=CHATBOT, dataset=et_test, api_url=API_URL_ET)
translate_to_file(source_language="et", dataset_type=TRAIN, dataset_name=CHATBOT, dataset=et_train,
                  api_url=API_URL_ET)

translate_to_file(source_language="lt", dataset_type=TEST, dataset_name=CHATBOT, dataset=lt_test, api_url=API_URL_LT)
translate_to_file(source_language="lt", dataset_type=TRAIN, dataset_name=CHATBOT, dataset=lt_train,
                  api_url=API_URL_LT)

lv_test = get_source_text(TEST, "lv", WEBAPPS)
print(lv_test)
translate_to_file(source_language="lv", dataset_type=TEST, dataset_name=WEBAPPS, dataset=lv_test, api_url=API_URL_LV)

lv_train = get_source_text(TRAIN, "lv", WEBAPPS)
print(lv_train)
translate_to_file(source_language="lv", dataset_type=TRAIN, dataset_name=WEBAPPS, dataset=lv_train,
                  api_url=API_URL_LV)

ru_test = get_source_text(TEST, "ru", WEBAPPS)
print(ru_test)
translate_to_file(source_language="ru", dataset_type=TEST, dataset_name=WEBAPPS, dataset=ru_test, api_url=API_URL_RU)

ru_train = get_source_text(TRAIN, "ru", WEBAPPS)
print(ru_train)
translate_to_file(source_language="ru", dataset_type=TRAIN, dataset_name=WEBAPPS, dataset=ru_train,
                  api_url=API_URL_RU)

et_test = get_source_text(TEST, "et", UBUNTU)
print(et_test)
translate_to_file(source_language="et", dataset_type=TEST, dataset_name=UBUNTU, dataset=et_test, api_url=API_URL_ET)

et_train = get_source_text(TRAIN, "et", UBUNTU)
print(et_train)
translate_to_file(source_language="et", dataset_type=TRAIN, dataset_name=UBUNTU, dataset=et_train, api_url=API_URL_ET)

lt_test = get_source_text(TEST, "lt", UBUNTU)
print(lt_test)
translate_to_file(source_language="lt", dataset_type=TEST, dataset_name=UBUNTU, dataset=lt_test, api_url=API_URL_LT)

lt_train = get_source_text(TRAIN, "lt", UBUNTU)
print(lt_train)
translate_to_file(source_language="lt", dataset_type=TRAIN, dataset_name=UBUNTU, dataset=lt_train, api_url=API_URL_LT)
