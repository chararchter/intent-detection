from typing import List

from transformers import pipeline


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


def translate_to_file(dataset_type: str, source_language: str, dataset: List[str], model_name: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param dataset_type: "test" or "train"
    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset: dataset of source_language and type e.g. "lv_train"
    :param model_name: model name e.g. "opus-mt-tc-big-et-en"
    """
    pipe = pipeline("translation", model=model_name)
    for line in dataset:
        print(line)
        output = pipe(line)
        print(output)
        if "error" in output:
            print(output)
            write_to_file(source_language, dataset_type, "error, original line:" + line)
        else:
            write_to_file(source_language, dataset_type, output[0]["translation_text"])


def write_to_file(source_language: str, dataset_type: str, output: str):
    """ Write the translated text to file.
    utf-8 encoding is specified in case the source text wasn't translated and still has the source language characters.

    :param source_language: "lv", "ru", "et" or "lt"
    :param dataset_type: "test" or "train"
    :param output: translated text or error and sentence in original language
    """
    with open(f"{source_language}_{dataset_type}.txt", "a", encoding="utf-8") as f:
        f.write(output + "\n")


def translate_to_english(dataset: List[str], model_name: str, source_language: str, dataset_type: str):
    pipe = pipeline("translation", model=model_name)
    translated_text = pipe(dataset)
    print(translated_text)
    write_to_file(source_language, dataset_type, translated_text)


lv_test = get_source_text("test", "lv")
ru_test = get_source_text("test", "ru")
et_test = get_source_text("test", "et")
lt_test = get_source_text("test", "lt")

lv_train = get_source_text("train", "lv")
ru_train = get_source_text("train", "ru")
et_train = get_source_text("train", "et")
lt_train = get_source_text("train", "lt")

translate_to_file(source_language="ru", dataset_type="test", dataset=ru_test, model_name=".\opus-mt-ru-en")
translate_to_file(source_language="ru", dataset_type="train", dataset=ru_train, model_name=".\opus-mt-ru-en")

translate_to_file(source_language="et", dataset_type="test", dataset=et_test, model_name=".\opus-mt-tc-big-et-en")
translate_to_file(source_language="et", dataset_type="train", dataset=et_train, model_name=".\opus-mt-tc-big-et-en")
