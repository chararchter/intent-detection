from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Conv1D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
        for line in list(f):
            array.append(line.split('\n')[0])
        return array


def get_source_text(
        dataset_type: str,
        source_language: str = None,
        labels: bool = False,
        machine_translated: bool = False
) -> List[str]:
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


# use keras.to_categorical() instead
def encode_labels(answers: List) -> List:
    """ Encode labels in one hot-encoding
    'FindConnection' corresponds to [[1, 0]]
    'DepartureTime' corresponds to [[0, 1]]
    """
    y = []
    for answer in answers:
        if answer == 'FindConnection':
            y.append([[1, 0]])
        else:
            y.append([[0, 1]])
    return y


def split_train_data(x, y, validation_size: int = 0.2):
    """ Split training set in training and validation
    :param x: data
    :param y: labels
    :param validation_size: what fraction of data to allocate to validation?
    :return:
    """
    return train_test_split(x, y, test_size=validation_size, stratify=y, random_state=42)


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
    # plt.show()


def get_word_embeddings(data: list, sentence_length: int):
    encoded_input = tokenizer(data, padding='max_length', max_length=sentence_length, truncation=True,
                              return_tensors='tf')
    return model_bert(encoded_input)["last_hidden_state"]


def training(train_dataset, train_answers, dataset_name: str, learning_rate: int, sentence_length: int, batch_size: int, epochs: int):

    # Split training dataset in training data and validation data
    train_data, validation_data, train_labels, validation_labels = split_train_data(train_dataset, train_answers)

    # Convert data to embeddings
    train_data = get_word_embeddings(train_data, sentence_length)
    validation_data = get_word_embeddings(validation_data, sentence_length)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    validation_labels_encoded = label_encoder.transform(validation_labels)

    # Convert integer labels to categorical data
    num_classes = len(label_encoder.classes_)
    print(f"num_classes {num_classes}")  # 2
    train_labels_categorical = to_categorical(train_labels_encoded, num_classes=num_classes)
    validation_labels_categorical = to_categorical(validation_labels_encoded, num_classes=num_classes)

    print(f"train_data.shape {train_data.shape}")  # should print (num_samples, sentence_length, hidden_size) (80, 20, 768)
    print(f"train_labels_categorical.shape {train_labels_categorical.shape}")  # should print (num_samples, num_classes) (80, 2)
    print(f"validation_labels_categorical.shape {validation_labels_categorical.shape}")  # should print (num_samples, num_classes) (20, 2)

    classification_model = get_classification_model(learning_rate, sentence_length)

    history = classification_model.fit(
        train_data,
        y=train_labels_categorical,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_data, validation_labels_categorical)
    )

    plot_performance(history.history['accuracy'], dataset=dataset_name, x_label='accuracy')
    plot_performance(history.history['loss'], dataset=dataset_name, x_label='loss')

    return classification_model


def test_classification_model(classification_model, dataset, encoded_test_labels, batch_size, sentence_length) -> float:
    encoded_input = tokenizer(
        dataset,
        padding='max_length',
        max_length=sentence_length,
        truncation=True,
        return_tensors='tf'
    )
    classification_input = model_bert(encoded_input)["last_hidden_state"]

    test_loss, test_accuracy = classification_model.evaluate(classification_input, encoded_test_labels,
                                                             batch_size=batch_size)
    print('Test Loss: {:.2f}'.format(test_loss))
    print('Test Accuracy: {:.2f}'.format(test_accuracy))
    return test_accuracy
