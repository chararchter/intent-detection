from typing import Iterable, Tuple

import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.framework.ops import EagerTensor

from model import training, \
    test_classification_model, get_source_text, split_train_data, get_embeddings_tokenizer_model


class MyModel:
    def __init__(self, batch_size: int, learning_rate: float, epochs: int, sentence_length: int, model_name: str,
                 languages=("en", "lv", "ru", "et", "lt")):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sentence_length = sentence_length
        self.model_name = model_name
        self.data = dict()
        self.datasets = dict()
        self.results = pd.DataFrame()
        self.units = 2
        self.hidden_size = 768
        # allows to run model one language at the time
        self.languages = [languages] if isinstance(languages, str) else languages
        self.non_eng_languages = list(set(self.languages) - {"en"})
        self.non_eng_languages = [language + "_en" for language in self.non_eng_languages]
        self.csv_file_name = f"{self.model_name}_results.csv"

        self.tokenizer, self.model = get_embeddings_tokenizer_model(self.model_name)

        self.init_dataset()
        self.init_data()
        self.init_results()

    def init_dataset(self):
        self.datasets = {
            "test": self.languages,
            "train": self.languages
        }

    def init_results(self):
        self.results['hyperparameters'] = [self.model_name, self.batch_size, self.sentence_length, self.learning_rate,
                                           self.epochs]

    def init_data(self):
        self.get_dataset()

        self.labels_to_categorical()

        self.split_validation()

        self.convert_to_embeddings()

        self.merge_all_data('train_all', ['train_en', 'train_lv', 'train_ru', 'train_et', 'train_lt'])
        self.merge_all_data('train_all_labels',
                            ['train_en_labels', 'train_lv_labels', 'train_ru_labels', 'train_et_labels',
                             'train_lt_labels'])
        self.merge_all_data('train_all_validation',
                            ['train_en_validation', 'train_lv_validation', 'train_ru_validation', 'train_et_validation',
                             'train_lt_validation'])
        self.merge_all_data('train_all_labels_validation', ['train_en_labels_validation', 'train_lv_labels_validation',
                                                            'train_ru_labels_validation', 'train_et_labels_validation',
                                                            'train_lt_labels_validation'])

        self.merge_all_data('train_all_en', ['train_en', 'train_lv_en', 'train_ru_en', 'train_et_en', 'train_lt_en'])
        self.merge_all_data('train_all_en_labels',
                            ['train_en_labels', 'train_lv_en_labels', 'train_ru_en_labels', 'train_et_en_labels',
                             'train_lt_en_labels'])
        self.merge_all_data('train_all_en_validation',
                            ['train_en_validation', 'train_lv_en_validation', 'train_ru_en_validation',
                             'train_et_en_validation',
                             'train_lt_en_validation'])
        self.merge_all_data('train_all_en_labels_validation',
                            ['train_en_labels_validation', 'train_lv_en_labels_validation',
                             'train_ru_en_labels_validation',
                             'train_et_en_labels_validation', 'train_lt_en_labels_validation'])

    def is_translated(self, translated: bool = False) -> Tuple[list, list, str, str]:
        """ Return appropriate parameters based on whether the data has been machine translated to English
        :param translated: has the data been machine translated to English?
        :return: which list of languages to use, initialized result set and column name for results dataframe
        """
        if translated:
            return self.non_eng_languages, [None], "translated", "_en"
        else:
            return self.languages, [], "untranslated", ""

    def train_and_test_on_same_language(self, translated: bool = False):
        """ Each language has its own model, e.g., training on Latvian, testing on Latvian
        :param translated: has the data been machine translated to English?
        """
        temp_languages, temp_results, col_name, discard = self.is_translated(translated)
        for language in temp_languages:
            classification = training(self.data, language, self.learning_rate, self.sentence_length, self.batch_size,
                                      self.epochs, self.model_name)
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results[f"1_{col_name}"] = temp_results
        self.results.to_csv(self.csv_file_name, index=False)

    def train_on_all_languages_test_on_one(self, translated: bool = False):
        """ One model trained on all language datasets, tested on each language separately
        e.g., training on all, testing on Latvian
        :param translated: has the data been machine translated to English?
        """
        temp_languages, temp_results, col_name, identifier = self.is_translated(translated)
        classification = training(self.data, f"all{identifier}", self.learning_rate, self.sentence_length,
                                  self.batch_size, self.epochs, self.model_name)

        for language in temp_languages:
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results[f"2_{col_name}"] = temp_results
        self.results.to_csv(self.csv_file_name, index=False)

    def train_on_english_test_on_non_english(self, translated: bool = False):
        """ Trained on English only, tested on non-English
        e.g., training on English, testing on Latvian
        :param translated: has the data been machine translated to English?
        """
        temp_languages, temp_results, col_name, discard = self.is_translated(translated)
        classification = training(self.data, "en", self.learning_rate, self.sentence_length,
                                  self.batch_size, self.epochs, self.model_name)

        for language in temp_languages:
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results[f"3_{col_name}"] = temp_results
        self.results.to_csv(self.csv_file_name, index=False)

    def get_dataset(self):
        """ Initialize data dictionary with values read from files
        """
        for key, value in self.datasets.items():
            self.data.update({f"{key}_labels": get_source_text(dataset_type=key, labels=True)})
            for lang in value:
                self.data.update({f"{key}_{lang}": get_source_text(dataset_type=key, source_language=lang)})
                if lang != "en":
                    self.data.update({f"{key}_{lang}_en": get_source_text(
                        dataset_type=key,
                        source_language=lang,
                        machine_translated=True
                    )})

    def labels_to_categorical(self):
        """ Convert string labels to categorical data
        """
        label_encoder = LabelEncoder()
        for key in self.data.keys():
            if "labels" in key:
                # Encode string labels to integer labels
                self.data[key] = label_encoder.fit_transform(self.data[key])
                # Convert integer labels to categorical data
                self.data[key] = to_categorical(self.data[key], num_classes=len(label_encoder.classes_))

    def split_validation(self):
        """ Split training dataset in training and validation
        """
        for key, value in self.datasets.items():
            if key == "train":
                for lang in value:
                    self.data[f"{key}_{lang}"], \
                        self.data[f"{key}_{lang}_validation"], \
                        self.data[f"{key}_{lang}_labels"], \
                        self.data[f"{key}_{lang}_labels_validation"] = split_train_data(self.data[f"{key}_{lang}"],
                                                                                        self.data[f"{key}_labels"])
                    if lang != "en":
                        self.data[f"{key}_{lang}_en"], \
                            self.data[f"{key}_{lang}_en_validation"], \
                            self.data[f"{key}_{lang}_en_labels"], \
                            self.data[f"{key}_{lang}_en_labels_validation"] = split_train_data(
                            self.data[f"{key}_{lang}_en"],
                            self.data[f"{key}_labels"])

    def convert_to_embeddings(self):
        """ Loop through the data dictionary and call get_word_embeddings on each key that isn't a label
        """
        for key, value in self.data.items():
            if "labels" not in key:
                self.data[key] = self.get_word_embeddings(value)

    def get_word_embeddings(self, vectorizable_strings: list) -> EagerTensor:
        """ Convert input to word embeddings
        """

        encoded_input = self.tokenizer(
            vectorizable_strings,
            padding='max_length',
            max_length=self.sentence_length,
            truncation=True,
            return_tensors='tf'
        )
        return self.model(encoded_input)["last_hidden_state"]

    def merge_all_data(self, new_key_name: str, keys_to_merge: Iterable):
        new_values = tf.concat([self.data[key] for key in keys_to_merge], axis=0)
        self.data.update({new_key_name: new_values})


# learning_rate=0.0001 is too small
if __name__ == "__main__":
    # model = MyModel(batch_size=24, learning_rate=0.001, epochs=100, sentence_length=20, model_name="xlm-roberta-base")
    # model.train_and_test_on_same_language(translated=True)
    # model.train_and_test_on_same_language(translated=False)
    # model.train_on_all_languages_test_on_one(translated=True)
    # model.train_on_all_languages_test_on_one(translated=False)
    # model.train_on_english_test_on_non_english(translated=True)
    # model.train_on_english_test_on_non_english(translated=False)

    model = MyModel(batch_size=24, learning_rate=0.00001, epochs=100, sentence_length=20,
                    model_name="bert-base-multilingual-cased")
    model.train_and_test_on_same_language(translated=True)
    # model.train_and_test_on_same_language(translated=False)
    # model.train_on_all_languages_test_on_one(translated=True)
    # model.train_on_all_languages_test_on_one(translated=False)
    # model.train_on_english_test_on_non_english(translated=True)
    # model.train_on_english_test_on_non_english(translated=False)
