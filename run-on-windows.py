from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv1D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel


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
        for line in list(f):
            array.append(line.split('\n')[0])
        return array


def get_source_text(dataset_type: str, source_language: str = None, labels: bool = False, machine_translated: bool = False) -> List[str]:
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


# Read the NLU-datasets in their original source languages

en_test = get_source_text("test", "en")
lv_test = get_source_text("test", "lv")
ru_test = get_source_text("test", "ru")
et_test = get_source_text("test", "et")
lt_test = get_source_text("test", "lt")

en_train = get_source_text("train", "en")
lv_train = get_source_text("train", "lv")
ru_train = get_source_text("train", "ru")
et_train = get_source_text("train", "et")
lt_train = get_source_text("train", "lt")

train_answers = get_source_text(dataset_type="train", labels=True)
test_answers = get_source_text(dataset_type="test", labels=True)

assert len(train_answers) == len(en_train)

# Read non-English NLU-datasets that have been pre-machine-translated to English

lv_test_en = get_source_text("test", "lv", machine_translated=True)
ru_test_en = get_source_text("test", "ru", machine_translated=True)
et_test_en = get_source_text("test", "et", machine_translated=True)
lt_test_en = get_source_text("test", "lt", machine_translated=True)

lv_train_en = get_source_text("train", "lv", machine_translated=True)
ru_train_en = get_source_text("train", "ru", machine_translated=True)
et_train_en = get_source_text("train", "et", machine_translated=True)
lt_train_en = get_source_text("train", "lt", machine_translated=True)

print(lv_test[0])
print(lv_test_en[0])

# # Definitions
# ## Model and tokenizer


model_name = "bert-base-multilingual-cased"  # loading from huggingface
model_name = "./bert-base-multilingual-cased"  # loading from local path

tokenizer = BertTokenizer.from_pretrained(model_name)
model_bert = TFBertModel.from_pretrained(model_name)


# ## Labels


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


encoded_train_labels = encode_labels(train_answers)
encoded_train_labels = tf.convert_to_tensor(encoded_train_labels)

encoded_test_labels = encode_labels(test_answers)
encoded_test_labels = tf.convert_to_tensor(encoded_test_labels)

print(encoded_train_labels)


def split_train_data(x, y, validation_size: int = 0.2):
    """ Split training set in training and validation
    :param x: data
    :param y: labels
    :param validation_size: what fraction of data to allocate to validation?
    :return:
    """
    return train_test_split(x, y, test_size=validation_size, stratify=y, random_state=42)


train_en, validation_en, train_labels, validation_labels = split_train_data(en_train, train_answers)

assert len(train_en) + len(validation_en) == len(en_train)
assert len(train_labels) + len(validation_labels) == len(train_answers)

# Count occurrences of labels in training dataset
unique, counts = np.unique(np.array(train_labels), return_counts=True)
dict(zip(unique, counts))

# Count occurrences of labels in training dataset
unique, counts = np.unique(np.array(validation_labels), return_counts=True)
dict(zip(unique, counts))


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