import json
import os
from pathlib import Path
from typing import List

import requests

from secret_api_token import API_TOKEN

API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tc-big-et-en"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def translate(source_text: List[str]) -> List[str]:
    """ Query each element of the array
    """
    target_text = []

    for line in source_text:
        translated_line = query(line)
        print(translated_line)
        target_text.append(translated_line)
    return target_text


def get_data(path: str) -> List[str]:
    """ Read text file in the specified path and append each line without \n as an element to an array.
    Encoding is specified to correctly read files in Russian.

    :param path: path e.g. "chatbot\chatbot_train_ans.txt"
    :return: array e.g. ['FindConnection', 'FindConnection', ..., 'FindConnection']
    """
    with open(path, encoding='utf-8') as f:
        array = []
        for line in list(f):
            # print(line)
            print(line.split('\n')[0])
            array.append(line.split('\n')[0])
        return array


def read_source_text() -> List[str]:
    """ Iterate through files in path_list and read their contents
    :return: arrays of test and train datasets for each language
    """
    if "NLU-datasets" not in os.getcwd():
        os.chdir("./NLU-datasets")

    path_list = Path("chatbot").glob("**/*.txt")

    for path in path_list:
        # because path is object not string
        path_in_str = str(path)
        # print(path_in_str)
        # if path_in_str == "chatbot\lv\chatbot_test_q.txt":
        #     lv_test = get_data(path_in_str)
        # elif path_in_str == "chatbot\lv\chatbot_train_q.txt":
        #     lv_train = get_data(path_in_str)
        # elif path_in_str == "chatbot\\ru\chatbot_test_q.txt":
        #     ru_test = get_data(path_in_str)
        # elif path_in_str == "chatbot\\ru\chatbot_train_q.txt":
        #     ru_train = get_data(path_in_str)
        if path_in_str == "chatbot\et\chatbot_test_q.txt":
            et_test = get_data(path_in_str)
        # elif path_in_str == "chatbot\et\chatbot_train_q.txt":
        #     et_train = get_data(path_in_str)
        # elif path_in_str == "chatbot\lt\chatbot_test_q.txt":
        #     lt_test = get_data(path_in_str)
        # elif path_in_str == "chatbot\lt\chatbot_train_q.txt":
        #     lt_train = get_data(path_in_str)

    if "NLU-datasets" in os.getcwd():
        os.chdir("..")

    # return lv_test, lv_train, ru_test, ru_train, et_test, et_train, lt_test, lt_train
    return et_test


def write_to_file(source_language: str, dataset_type: str, translated_text: List[dict]):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset_type: "test" or "train"
    :param translated_text: array of arrays with one dictionary where key='translation_text' and value is the translated text
    e.g. [[{'translation_text': "Taxi's waiting."}]]
    """
    with open(f"{source_language}_{dataset_type}.txt", "w", encoding="utf-8") as f:
        for line in translated_text:
            f.write(line[0]["translation_text"] + "\n")


def translate_to_english(dataset: List[str], source_language: str, dataset_type: str):
    translated_text = translate(dataset)
    write_to_file(source_language, dataset_type, translated_text)


et_test = read_source_text()


def translate_to_english(dataset: List[str], source_language: str, dataset_type: str):
    translated_text = translate(dataset)
    write_to_file(source_language, dataset_type, translated_text)


translate_to_english(et_test, "et", "test")
# translate_to_english(et_train, "et", "train")
