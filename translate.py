from typing import List

from transformers import pipeline

from model import read_file


def get_source_text(dataset_type: str, source_language: str, dataset_name: str) -> List[str]:
    """ Wrapper for get_data that provides file path.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et", "lt"
    :param dataset_name: "chatbot", "askubuntu" or "webapps"
    :return: array of file contents for specified file
    """
    return read_file(f"NLU-datasets\{dataset_name}\{source_language}\{dataset_name}_{dataset_type}_q.txt")


def translate_to_file(dataset_type: str, source_language: str, dataset: List[str], dataset_name: str, model_name: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset: dataset of source_language and type e.g. "lv_train"
    :param dataset_name: "chatbot", "askubuntu" or "webapps"
    :param model_name: model name e.g. "opus-mt-tc-big-et-en"
    """
    pipe = pipeline("translation", model=model_name)
    for line in dataset:
        print(line)
        output = pipe(line)
        print(output)
        if "error" in output:
            print(output)
            write_to_file(source_language, dataset_type, dataset_name, "error, original line:" + line)
        else:
            write_to_file(source_language, dataset_type, dataset_name, output[0]["translation_text"])


def write_to_file(source_language: str, dataset_type: str, dataset_name: str, output: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset_type: "test" or "train"
    :param dataset_name: "chatbot", "askubuntu" or "webapps"
    :param output: translated text or error and sentence in original language
    """
    with open(f"{dataset_name}_{source_language}_{dataset_type}.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")


def translate_to_english(dataset: List[str], model_name: str, source_language: str, dataset_type: str):
    pipe = pipeline("translation", model=model_name)
    translated_text = pipe(dataset)
    print(translated_text)
    write_to_file(source_language, dataset_type, translated_text)


model = dict()
model = {
    "lv": ".\opus-mt-tc-big-lv-en",
    "ru": ".\opus-mt-ru-en",
    "et": ".\opus-mt-tc-big-et-en",
    "lt": ".\opus-mt-tc-big-lt-en",
}

for language, model_name in model.items():
    for dataset_name in ["chatbot", "webapps", "askubuntu"]:
        for dataset_type in ["test", "train"]:
            dataset = get_source_text(dataset_type, language, dataset_name)
            translate_to_file(
                source_language=language, dataset_type=dataset_type, dataset_name=dataset_name, dataset=dataset,
                model_name=model_name
            )
