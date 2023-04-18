from typing import Iterable, Literal

import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv1D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.framework.ops import EagerTensor

from model import training, \
    test_classification_model, get_source_text, split_train_data, tokenizer, model_bert


class MyModel:
    def __init__(self, batch_size: int, learning_rate: float, epochs: int, sentence_length: int,
                 languages=("en", "lv", "ru", "et", "lt")):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sentence_length = sentence_length
        self.data = dict()
        self.datasets = dict()
        self.results = pd.DataFrame()
        self.units = 2
        self.hidden_size = 768
        # allows to run model one language at the time
        self.languages = [languages] if isinstance(languages, str) else languages
        self.non_eng_languages = list(set(self.languages) - {"en"})
        self.non_eng_languages = [language + "_en" for language in self.non_eng_languages]

        self.init_dataset()
        self.init_data()
        self.init_results()

    def init_dataset(self):
        self.datasets = {
            "test": self.languages,
            "train": self.languages
        }

    def init_results(self):
        self.results['hyperparameters'] = [self.batch_size, self.sentence_length, self.learning_rate, self.epochs, None]

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

    def run_untranslated(self, translated: bool = False, trans: Literal["translated", "untranslated"] = "translated"):
        # data falls in two categories:
        # one is untranslated (original language) or translated (to english)
        # second one normal, all languages meshed up together or just english
        temp_results = []
        for language in self.languages:
            classification = training(self.data, language, self.learning_rate, self.sentence_length, self.batch_size,
                                      self.epochs)
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results['1_method'] = temp_results
        self.results.to_csv("results.csv", index=False)

    def run_translated(self):
        temp_results = [None]
        for language in self.non_eng_languages:
            classification = training(self.data, language, self.learning_rate, self.sentence_length,
                                      self.batch_size, self.epochs)
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results['2_method'] = temp_results
        self.results.to_csv("results.csv", index=False)

    def run_untranslated_together(self):
        classification = training(self.data, "all", self.learning_rate, self.sentence_length,
                                  self.batch_size, self.epochs)

        temp_results = []
        for language in self.languages:
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results['3_method'] = temp_results
        self.results.to_csv("results.csv", index=False)

    def run_translated_together(self):
        classification = training(self.data, "all_en", self.learning_rate, self.sentence_length,
                                  self.batch_size, self.epochs)

        temp_results = [None]
        for language in self.non_eng_languages:
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results['4_method'] = temp_results
        self.results.to_csv("results.csv", index=False)

    def train_on_english_only_untranslated(self):
        classification = training(self.data, "en", self.learning_rate, self.sentence_length,
                                  self.batch_size, self.epochs)
        temp_results = []
        for language in self.languages:
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results['5_method'] = temp_results
        self.results.to_csv("results.csv", index=False)

    def train_on_english_only_translated(self):
        classification = training(self.data, "en", self.learning_rate, self.sentence_length,
                                  self.batch_size, self.epochs)
        temp_results = [None]
        for language in self.non_eng_languages:
            temp_results.append(test_classification_model(classification, self.data, language, self.batch_size))
        self.results['6_method'] = temp_results
        self.results.to_csv("results.csv", index=False)

    def create_model_one_layer(self):
        model = Sequential()
        model.add(tf.keras.Input(shape=(self.sentence_length, self.hidden_size)))
        model.add(Dense(self.units, activation='softmax'))
        print(model.summary())
        model.add(Conv1D(self.units, self.sentence_length, padding="valid", activation="softmax"))
        print(model.summary())
        # model.add(MaxPooling1D(pool_size=2))
        # print(model.summary())
        model.add(Dropout(0.05))  # make smaller dropout
        print(model.summary())
        model.add(Dense(self.units, activation='softmax'))
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=1)))  # squeeze the output to remove dimension with size 1
        print(model.summary())
        return model

    def create_adam_optimizer(self, beta_1=0.9, beta_2=0.999, weight_decay=0, epsilon=0, amsgrad=False):
        # sgd is worse than adam
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                        amsgrad=amsgrad,
                                        weight_decay=weight_decay)

    def get_classification_model(self):
        optimizer = self.create_adam_optimizer()
        classification_model = self.create_model_one_layer()

        classification_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return classification_model

    def get_dataset(self):
        """ Initialize data dictionary with values read from files
        """
        for key, value in self.datasets.items():
            self.data.update({f"{key}_labels": get_source_text(dataset_type=key, labels=True)})
            for lang in value:
                self.data.update({f"{key}_{lang}": get_source_text(dataset_type=key, source_language=lang)})
                if lang != "en":
                    self.data.update({f"{key}_{lang}_en": get_source_text(dataset_type=key, source_language=lang,
                                                                          machine_translated=True)})

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
        encoded_input = tokenizer(vectorizable_strings, padding='max_length', max_length=self.sentence_length,
                                  truncation=True,
                                  return_tensors='tf')
        return model_bert(encoded_input)["last_hidden_state"]

    def merge_all_data(self, new_key_name: str, keys_to_merge: Iterable):
        new_values = tf.concat([self.data[key] for key in keys_to_merge], axis=0)
        self.data.update({new_key_name: new_values})


if __name__ == "__main__":
    model = MyModel(batch_size=52, learning_rate=0.003, epochs=100, sentence_length=20)
    model.run_untranslated()
    model.run_translated()
    model.run_untranslated_together()
    model.run_translated_together()
    model.train_on_english_only_untranslated()
    model.train_on_english_only_translated()
