import json
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


def translate_to_english(dataset: List[str], source_language: str, dataset_type: str):
    translated_text = translate(dataset)
    write_to_file(source_language, dataset_type, translated_text)


data = read_source_text("test", "et", False)
print(data)
