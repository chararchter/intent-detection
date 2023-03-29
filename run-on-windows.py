from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Conv1D

from model import tokenizer, model_bert


# reload function again without restarting kernel
# from importlib import reload
# import function
# reload(function)


# # Read the natural language understanding dataset and BERT model
# 
# Clone the repos inside intent-detection directory
# ```
# git clone https://github.com/tilde-nlp/NLU-datasets.git
# git clone https://huggingface.co/bert-base-multilingual-cased
# ```
# 
# Directory tree should be as follows
# ```
# /intent-detection
# ├── NLU-datasets
# ├── bert-base-multilingual-cased
# ├── run-on-windows.ipynb
# ```


def read_file(path: str) -> List[str]:
    """ Read path and append each line without \n as an element to an array.
    Encoding is specified to correctly read files in Russian.
    Example output: ['FindConnection', 'FindConnection', ..., 'FindConnection']
    """
    with open(path, encoding='utf-8') as f:
        array = []
        for line in f:
            array.append(line.rstrip("\n"))
        return array


def get_source_text(dataset_type: str, source_language: str = None, labels: bool = False,
                    machine_translated: bool = False) -> List[str]:
    """ Wrapper for read_file that provides file path.
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
        return read_file(f"NLU-datasets\chatbot\chatbot_{dataset_type}_ans.txt")
    elif machine_translated:
        return read_file(f"machine-translated-datasets\{source_language}_{dataset_type}.txt")
    else:
        return read_file(f"NLU-datasets\chatbot\{source_language}\chatbot_{dataset_type}_q.txt")


def get_dataset(datasets: dict, need_labels: bool = True) -> dict:
    """
    :param datasets:
    :param need_labels: redundant argument, i'm gonna remove it later
    :return: dictionary with dataset type, language and optional labels and '_en' as keys and list of input data as values
    """
    results = dict()
    for key, value in datasets.items():
        if need_labels:
            results.update({f"{key}_labels": get_source_text(dataset_type=key, labels=True)})
        for lang in value:
            results.update({f"{key}_{lang}": get_source_text(dataset_type=key, source_language=lang)})
            if lang != "en":
                results.update({f"{key}_{lang}_en": get_source_text(dataset_type=key, source_language=lang,
                                                                    machine_translated=True)})
    return results


def split_train_data(x: list, y: list, validation_size: int = 0.2):
    """ Split training set in training and validation
    :param x: data
    :param y: labels
    :param validation_size: what fraction of data to allocate to training?
    :return:
    """
    return train_test_split(x, y, test_size=validation_size, stratify=y, random_state=42)


def split_validation(datasets: dict, data: dict) -> dict:
    """ Split training dataset in training and validation
    :param datasets: dictionary with dataset type as key and list of languages as value
    :param data: dictionary with test/train, language and labels as key and data as values
    :return: updated data dictionary where each train key is split in train and validation
    """
    for key, value in datasets.items():
        if key == "train":
            for lang in value:
                data[f"{key}_{lang}"], \
                    data[f"{key}_{lang}_validation"], \
                    data[f"{key}_{lang}_labels"], \
                    data[f"{key}_{lang}_labels_validation"] = split_train_data(data[f"{key}_{lang}"],
                                                                               data[f"{key}_labels"])
                if lang != "en":
                    data[f"{key}_{lang}_en"], \
                        data[f"{key}_{lang}_en_validation"], \
                        data[f"{key}_{lang}_en_labels"], \
                        data[f"{key}_{lang}_en_labels_validation"] = split_train_data(data[f"{key}_{lang}_en"],
                                                                                      data[f"{key}_labels"])
    return data


def labels_to_categorical(data: dict) -> dict:
    """ Convert string labels to categorical data
    """
    label_encoder = LabelEncoder()
    for key in data.keys():
        if "labels" in key:
            # Encode string labels to integer labels
            data[key] = label_encoder.fit_transform(data[key])
            # Convert integer labels to categorical data
            data[key] = to_categorical(data[key], num_classes=len(label_encoder.classes_))
    return data


# Overview of the languages for each dataset type
languages = ["en", "lv", "ru", "et", "lt"]
datasets = {
    "test": languages,
    "train": languages
}

data = get_dataset(datasets)

data = labels_to_categorical(data)

data = split_validation(datasets, data)
print(data)

# ## Training


# it works worse with dropout

def create_model_one_layer(sentence_length: int, units: int = 2, hidden_size: int = 768):
    """
    returns <tf.Tensor: shape=(1, 1, units), dtype=float32>
    e.g. <tf.Tensor: shape=(1, 1, 2), dtype=float32>
    where 2 = units
    """
    model = Sequential()
    model.add(tf.keras.Input(shape=(sentence_length, hidden_size)))
    model.add(Dense(units, activation='softmax'))
    model.add(Conv1D(units, sentence_length, padding="valid", activation="softmax"))
    # model.add(MaxPooling1D(pool_size=2)) # enable this layer
    model.add(Dropout(0.1))  # make smaller dropout
    # model.add(Dense(units, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def create_adam_optimizer(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=0, epsilon=0, amsgrad=False):
    # sgd is worse than adam
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad,
                                    weight_decay=weight_decay)


def get_classification_model(learning_rate: int, sentence_length: int):
    optimizer = create_adam_optimizer(lr=learning_rate)
    classification_model = create_model_one_layer(sentence_length=sentence_length)

    classification_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return classification_model


def plot_performance(data, dataset: str, x_label: str = 'accuracy'):
    plt.plot(data)
    ax = plt.gca()
    ax.set_xlabel('epochs')
    ax.set_ylabel(x_label)
    plt.title(f"{dataset} model {x_label}")
    plt.savefig(f"graphs/{dataset}-{x_label}.png")
    # plt.savefig(f"{dataset}{x_label}.pdf", dpi=150) # pdf for LaTeX
    plt.show()


def get_word_embeddings(data: list, sentence_length: int):
    encoded_input = tokenizer(data, padding='max_length', max_length=sentence_length, truncation=True,
                              return_tensors='tf')
    return model_bert(encoded_input)["last_hidden_state"]


def training(train_dataset, dataset_name: str, learning_rate: int, sentence_length: int):
    train_data, validation_data, train_labels, validation_labels = split_train_data(train_dataset, train_answers)

    train_data = get_word_embeddings(train_data, sentence_length)
    validation_data = get_word_embeddings(validation_data, sentence_length)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    validation_labels_encoded = label_encoder.transform(validation_labels)
    print(train_labels_encoded)

    # Convert integer labels to categorical data
    num_classes = len(label_encoder.classes_)
    print(f"num_classes {num_classes}")  # 2
    train_labels_categorical = to_categorical(train_labels_encoded, num_classes=num_classes)
    validation_labels_categorical = to_categorical(validation_labels_encoded, num_classes=num_classes)

    print(
        f"train_data.shape {train_data.shape}")  # should print (num_samples, sentence_length, hidden_size) (80, 20, 768)
    print(
        f"train_labels_categorical.shape {train_labels_categorical.shape}")  # should print (num_samples, num_classes) (80, 2)
    print(
        f"validation_labels_categorical.shape {validation_labels_categorical.shape}")  # should print (num_samples, num_classes) (20, 2)

    classification_model = get_classification_model(learning_rate, sentence_length)

    history = classification_model.fit(train_data, y=train_labels_categorical, batch_size=batch_size,
                                       epochs=number_of_epochs,
                                       validation_data=(validation_data, validation_labels_categorical))

    plot_performance(history.history['accuracy'], dataset=dataset_name, x_label='accuracy')
    plot_performance(history.history['loss'], dataset=dataset_name, x_label='loss')

    return classification_model


# ## Test


def test_classification_model(classification_model, dataset, encoded_test_labels) -> float:
    encoded_input = tokenizer(dataset, padding='max_length', max_length=sentence_length, truncation=True,
                              return_tensors='tf')
    classification_input = model_bert(encoded_input)["last_hidden_state"]

    test_loss, test_accuracy = classification_model.evaluate(classification_input, encoded_test_labels,
                                                             batch_size=batch_size)
    print('Test Loss: {:.2f}'.format(test_loss))
    print('Test Accuracy: {:.2f}'.format(test_accuracy))
    return test_accuracy


# # A small example
# ## Sentence -> word embedding


batch_size = 4
sentence_length = 20

text = en_train[0:batch_size]
encoded_input = tokenizer(text, padding='max_length', max_length=sentence_length, truncation=True, return_tensors='tf')
encoded_input

# odict_keys(['last_hidden_state', 'pooler_output'])
inputs = model_bert(encoded_input)["last_hidden_state"]
inputs.shape

# ## Word embedding -> classification


units = 2
hidden_size = 768

optimizer = create_adam_optimizer(lr=0.03)

classification_model = Sequential()
classification_model.add(tf.keras.Input(shape=(sentence_length, hidden_size)))
classification_model.add(Dense(units, activation='softmax'))
classification_model.add(Conv1D(units, sentence_length, padding="valid", activation="softmax"))
classification_model.add(Dense(units, activation='softmax'))

classification_model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# initial probabilities
classification_model(inputs)

classification_model.fit(inputs, y=encoded_train_labels[0:batch_size], epochs=5)

# view the output of the classification_model: probabilities for labels

classification_model(inputs)

# # Methods

# ## Hyperparameters


batch_size = 25
sentence_length = 20
learning_rate = 0.0003
number_of_epochs = 100

len(en_train)

#

# ## Each language has its own model
# ### Using the original NLU-datasets
# #### Train


classification_model_en = training(en_train, dataset_name="en_train", learning_rate=learning_rate,
                                   sentence_length=sentence_length)

classification_model_lv = training(lv_train, dataset_name="lv_train", learning_rate=learning_rate,
                                   sentence_length=sentence_length, labels=encoded_train_labels)

classification_model_ru = training(ru_train, dataset_name="ru_train", learning_rate=learning_rate,
                                   sentence_length=sentence_length, labels=encoded_train_labels)

classification_model_et = training(et_train, dataset_name="et_train", learning_rate=learning_rate,
                                   sentence_length=sentence_length, labels=encoded_train_labels)

classification_model_lt = training(lt_train, dataset_name="lt_train", learning_rate=learning_rate,
                                   sentence_length=sentence_length, labels=encoded_train_labels)

# #### Test


accuracy_en = test_classification_model(classification_model_en, en_test, encoded_test_labels)

accuracy_lv = test_classification_model(classification_model_lv, lv_test, encoded_test_labels)

accuracy_ru = test_classification_model(classification_model_ru, ru_test, encoded_test_labels)

accuracy_et = test_classification_model(classification_model_et, et_test, encoded_test_labels)

accuracy_lt = test_classification_model(classification_model_lt, lt_test, encoded_test_labels)

df = pd.DataFrame()

df['1_method'] = [accuracy_en, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ### Using machine translated non-English datasets
# #### Train


classification_model_lv_en = training(lv_train_en, dataset_name="lv_train_en", learning_rate=learning_rate,
                                      sentence_length=sentence_length, labels=encoded_train_labels)

classification_model_ru_en = training(ru_train_en, dataset_name="ru_train_en", learning_rate=learning_rate,
                                      sentence_length=sentence_length, labels=encoded_train_labels)

classification_model_et_en = training(et_train_en, dataset_name="et_train_en", learning_rate=learning_rate,
                                      sentence_length=sentence_length, labels=encoded_train_labels)

classification_model_lt_en = training(lt_train_en, dataset_name="lt_train_en", learning_rate=learning_rate,
                                      sentence_length=sentence_length, labels=encoded_train_labels)

# #### Test


accuracy_lv = test_classification_model(classification_model_lv_en, lv_test_en, encoded_test_labels)

accuracy_ru = test_classification_model(classification_model_ru_en, ru_test_en, encoded_test_labels)

accuracy_et = test_classification_model(classification_model_et_en, et_test_en, encoded_test_labels)

accuracy_lt = test_classification_model(classification_model_lt_en, lt_test_en, encoded_test_labels)

df['2_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ## One model trained on all languages


# one big train label dataset
all_train_labels = []

for i in range(5):
    all_train_labels.extend(train_answers)

# one big test label dataset
all_test_labels = []

for i in range(5):
    all_test_labels.extend(test_answers)

all_train_labels = encode_labels(all_train_labels)
all_train_labels = tf.convert_to_tensor(all_train_labels)

all_test_labels = encode_labels(all_test_labels)
all_test_labels = tf.convert_to_tensor(all_test_labels)

# ### Using the original NLU-datasets


# one big training dataset
all_train = []
all_train.extend(en_train)
all_train.extend(lv_train)
all_train.extend(ru_train)
all_train.extend(et_train)
all_train.extend(lt_train)

classification_model_all = training(all_train, dataset_name="all_train", learning_rate=learning_rate,
                                    sentence_length=sentence_length, labels=all_train_labels)

accuracy_en = test_classification_model(classification_model_all, en_test, encoded_test_labels)

accuracy_lv = test_classification_model(classification_model_all, lv_test, encoded_test_labels)

accuracy_ru = test_classification_model(classification_model_all, ru_test, encoded_test_labels)

accuracy_et = test_classification_model(classification_model_all, et_test, encoded_test_labels)

accuracy_lt = test_classification_model(classification_model_all, lt_test, encoded_test_labels)

df['3_method'] = [accuracy_en, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ### Using machine translated non-English datasets


# one big training dataset
all_train_en = []
all_train_en.extend(en_train)
all_train_en.extend(lv_train_en)
all_train_en.extend(ru_train_en)
all_train_en.extend(et_train_en)
all_train_en.extend(lt_train_en)

classification_model_all = training(all_train_en, dataset_name="all_train_en", learning_rate=learning_rate,
                                    sentence_length=sentence_length, labels=all_train_labels)

accuracy_lv = test_classification_model(classification_model_all, lv_test, encoded_test_labels)

accuracy_ru = test_classification_model(classification_model_all, ru_test, encoded_test_labels)

accuracy_et = test_classification_model(classification_model_all, et_test, encoded_test_labels)

accuracy_lt = test_classification_model(classification_model_all, lt_test, encoded_test_labels)

df['4_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ## Trained only on English data


classification_model_en = training(en_train, dataset_name="en_train", learning_rate=learning_rate,
                                   sentence_length=sentence_length, labels=encoded_train_labels)

# ### Test on non-English data


accuracy_lv = test_classification_model(classification_model_en, lv_test, encoded_test_labels)

accuracy_ru = test_classification_model(classification_model_en, ru_test, encoded_test_labels)

accuracy_et = test_classification_model(classification_model_en, et_test, encoded_test_labels)

accuracy_lt = test_classification_model(classification_model_en, lt_test, encoded_test_labels)

df['5_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ### Test on non-English machine translated to English data


accuracy_lv = test_classification_model(classification_model_en, lv_test_en, encoded_test_labels)

accuracy_ru = test_classification_model(classification_model_en, ru_test_en, encoded_test_labels)

accuracy_et = test_classification_model(classification_model_en, et_test_en, encoded_test_labels)

accuracy_lt = test_classification_model(classification_model_en, lt_test_en, encoded_test_labels)

df['6_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

df.to_csv("many_models.csv")
