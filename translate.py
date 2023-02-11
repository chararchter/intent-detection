import os
from pathlib import Path
from typing import List

from transformers import pipeline


def get_data(path: str) -> List[str]:
    """ Read text file in the specified path and append each line without \n as an element to an array.
    Encoding is specified to correctly read files in Russian.

    :param path: path e.g. "chatbot\chatbot_train_ans.txt"
    :return: array e.g. ['FindConnection', 'FindConnection', ..., 'FindConnection']
    """
    with open(path, encoding='utf-8') as f:
        array = []
        for line in list(f):
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
        if path_in_str == "chatbot\lv\chatbot_test_q.txt":
            lv_test = get_data(path_in_str)
        elif path_in_str == "chatbot\lv\chatbot_train_q.txt":
            lv_train = get_data(path_in_str)
        elif path_in_str == "chatbot\\ru\chatbot_test_q.txt":
            ru_test = get_data(path_in_str)
        elif path_in_str == "chatbot\\ru\chatbot_train_q.txt":
            ru_train = get_data(path_in_str)
        elif path_in_str == "chatbot\et\chatbot_test_q.txt":
            et_test = get_data(path_in_str)
        elif path_in_str == "chatbot\et\chatbot_train_q.txt":
            et_train = get_data(path_in_str)
        elif path_in_str == "chatbot\lt\chatbot_test_q.txt":
            lt_test = get_data(path_in_str)
        elif path_in_str == "chatbot\lt\chatbot_train_q.txt":
            lt_train = get_data(path_in_str)

    if "NLU-datasets" in os.getcwd():
        os.chdir("..")

    return lv_test, lv_train, ru_test, ru_train, et_test, et_train, lt_test, lt_train


def write_to_file(source_language: str, dataset_type: str, translated_text: List[dict]):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset_type: "test" or "train"
    :param translated_text: array of dictionaries where key='translation_text' and value is the translated text
    e.g. [{'translation_text': "Taxi's waiting."}]
    """
    with open(f"{source_language}_{dataset_type}.txt", "w", encoding="utf-8") as f:
        for line in translated_text:
            f.write(line["translation_text"] + "\n")


def translate_to_english(dataset: List[str], model_name: str, source_language: str, dataset_type: str):
    pipe = pipeline("translation", model=model_name)
    translated_text = pipe(dataset)
    print(translated_text)
    write_to_file(source_language, dataset_type, translated_text)


lv_test, lv_train, ru_test, ru_train, et_test, et_train, lt_test, lt_train = read_source_text()

translate_to_english(et_test, "Helsinki-NLP/opus-mt-tc-big-et-en", "et", "test")
