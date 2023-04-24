from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Conv1D, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

model_name = "bert-base-multilingual-cased"  # loading from huggingface
model_name = "./bert-base-multilingual-cased"  # loading from local path

tokenizer = BertTokenizer.from_pretrained(model_name)
model_bert = TFBertModel.from_pretrained(model_name)


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


def plot_performance(training_data, validation_data, dataset: str, x_label: str = 'accuracy'):
    plt.plot(training_data, label='training')
    plt.plot(validation_data, label='validation')
    ax = plt.gca()
    ax.set_xlabel('epochs')
    ax.set_ylabel(x_label)
    plt.title(f"{dataset} model {x_label}")
    plt.legend(loc="center")
    plt.savefig(f"graphs/{dataset}-{x_label}.png")
    plt.show()


# def create_model(sentence_length: int, units: int = 2, hidden_size: int = 768):
#     model = Sequential()
#     model.add(tf.keras.Input(shape=(sentence_length, hidden_size)))
#     model.add(Dense(units, activation='softmax'))
#     print(model.summary())
#     model.add(Conv1D(units, sentence_length, padding="valid", activation="softmax"))
#     print(model.summary())
#     # model.add(MaxPooling1D(pool_size=2))
#     # print(model.summary())
#     model.add(Dropout(0.05))  # make smaller dropout
#     print(model.summary())
#     model.add(Dense(units, activation='softmax'))
#     model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))) # squeeze the output to remove dimension with size 1
#     print(model.summary())
#     return model

def create_model(sentence_length: int, units: int = 2, hidden_size: int = 768):
    kernel_regularizer_coef = 0.1
    model = Sequential()
    model.add(tf.keras.Input(shape=(sentence_length, hidden_size)))
    model.add(Dense(units, activation='softmax', kernel_regularizer=l2(kernel_regularizer_coef)))
    print(model.summary())
    model.add(Conv1D(units, sentence_length, padding="valid", activation="softmax",
                     kernel_regularizer=l2(kernel_regularizer_coef)))
    print(model.summary())
    # model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.05))
    print(model.summary())
    model.add(Dense(units, activation='softmax', kernel_regularizer=l2(kernel_regularizer_coef)))
    # squeeze the output to remove dimension with size 1
    model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1)))
    print(model.summary())
    return model


def create_adam_optimizer(lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=0, epsilon=0, amsgrad=False):
    # sgd is worse than adam
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad,
                                    weight_decay=weight_decay)


def get_classification_model(learning_rate: float, sentence_length: int):
    optimizer = create_adam_optimizer(lr=learning_rate)
    classification_model = create_model(sentence_length=sentence_length)

    classification_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return classification_model


def training(data, lang: str, learning_rate: float, sentence_length: int, batch_size: int, epochs: int):

    train_data = data[f"train_{lang}"]
    # TODO: stack attributes in different levels: test/train, language and machine translated yes/no
    # t = data["train"][lang][[identifier]]
    train_labels = data[f"train_{lang}_labels"]
    validation_data = data[f"train_{lang}_validation"]
    validation_labels = data[f"train_{lang}_labels_validation"]

    print(f"train_data.shape {train_data.shape}")  # (num_samples, sentence_length, hidden_size) (80, 20, 768)
    print(f"validation_data.shape {validation_data.shape}")  # (num_samples, sentence_length, hidden_size) (80, 20, 768)
    print(f"train_labels.shape {train_labels.shape}")  # (num_samples, num_classes) (80, 2)
    print(f"validation_labels.shape {validation_labels.shape}")  # (num_samples, num_classes) (20, 2)

    classification_model = get_classification_model(learning_rate, sentence_length)

    history = classification_model.fit(
        train_data,
        y=train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_data, validation_labels)
    )

    plot_performance(
        history.history['accuracy'],
        history.history['val_accuracy'],
        dataset=f"train_{lang}",
        x_label='accuracy'
    )

    plot_performance(
        history.history['loss'],
        history.history['val_loss'],
        dataset=f"train_{lang}",
        x_label='loss'
    )

    return classification_model


def test_classification_model(model, data: dict, lang: str, batch_size: int) -> float:

    test_data = data[f"train_{lang}"]
    test_labels = data[f"train_{lang}_labels"]

    test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print('Test Loss: {:.2f}'.format(test_loss))
    print('Test Accuracy: {:.2f}'.format(test_accuracy))
    return test_accuracy
