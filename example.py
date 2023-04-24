from typing import List

import tensorflow as tf
from keras.layers import Dense, Conv1D
from keras.models import Sequential

from model import tokenizer, model_bert, get_source_text, create_adam_optimizer

# Hyperparameters
units = 2
hidden_size = 768
batch_size = 4
sentence_length = 20
learning_rate = 0.03
epochs = 5


# for backwards compatibility, the new version is using keras.to_categorical()
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


# # A small example
# ## Sentence -> word embedding

en_train = get_source_text("train", "en")
train_answers = get_source_text(dataset_type="train", labels=True)

encoded_train_labels = encode_labels(train_answers)
encoded_train_labels = tf.convert_to_tensor(encoded_train_labels)

text = en_train[0:batch_size]
labels = encoded_train_labels[0:batch_size]

encoded_input = tokenizer(text, padding='max_length', max_length=sentence_length, truncation=True, return_tensors='tf')
print(encoded_input)

inputs = model_bert(encoded_input)["last_hidden_state"]
print(inputs.shape)


# ## Word embedding -> classification

optimizer = create_adam_optimizer(lr=learning_rate)

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
print(classification_model(inputs))

classification_model.fit(inputs, y=labels, epochs=epochs)

# view the output of the classification_model: probabilities for labels
print(classification_model(inputs))
