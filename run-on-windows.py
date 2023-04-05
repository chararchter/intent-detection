import pandas as pd
import tensorflow as tf

from model import get_dataset, labels_to_categorical, split_validation, training, convert_to_embeddings, \
    test_classification_model

# ## Hyperparameters

batch_size = 25
sentence_length = 20
learning_rate = 0.003
epochs = 100

# Overview of the languages for each dataset type
languages = ["en", "lv", "ru", "et", "lt"]
datasets = {
    "test": languages,
    "train": languages
}

data = get_dataset(datasets)

data = labels_to_categorical(data)

data = split_validation(datasets, data)

data = convert_to_embeddings(data, sentence_length)
print(data)
print(data.keys())

# ## Each language has its own model
# ### Using the original NLU-datasets
# #### Train

classification_en = training(data, "en", learning_rate, sentence_length, batch_size, epochs)

classification_lv = training(data, "lv", learning_rate, sentence_length, batch_size, epochs)

classification_ru = training(data, "ru", learning_rate, sentence_length, batch_size, epochs)

classification_et = training(data, "et", learning_rate, sentence_length, batch_size, epochs)

classification_lt = training(data, "lt", learning_rate, sentence_length, batch_size, epochs)

#### Test


accuracy_en = test_classification_model(classification_en, data, "en", batch_size)

accuracy_lv = test_classification_model(classification_lv, data, "lv", batch_size)

accuracy_ru = test_classification_model(classification_ru, data, "ru", batch_size)

accuracy_et = test_classification_model(classification_et, data, "et", batch_size)

accuracy_lt = test_classification_model(classification_lt, data, "lt", batch_size)

df = pd.DataFrame()

df['hyperparameters'] = [batch_size, sentence_length, learning_rate, epochs, None]

df['1_method'] = [accuracy_en, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
# print(df)

# ### Using machine translated non-English datasets
# #### Train


classification_lv_en = training(data, "lv_en", learning_rate, sentence_length, batch_size, epochs)

classification_ru_en = training(data, "ru_en", learning_rate, sentence_length, batch_size, epochs)

classification_et_en = training(data, "et_en", learning_rate, sentence_length, batch_size, epochs)

classification_lt_en = training(data, "lt_en", learning_rate, sentence_length, batch_size, epochs)

# #### Test

accuracy_lv = test_classification_model(classification_lv_en, data, "lv_en", batch_size)

accuracy_ru = test_classification_model(classification_ru_en, data, "ru_en", batch_size)

accuracy_et = test_classification_model(classification_et_en, data, "et_en", batch_size)

accuracy_lt = test_classification_model(classification_lt_en, data, "lt_en", batch_size)

df['2_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ## One model trained on all languages
# ### Using the original NLU-datasets

# One big training dataset

train_all = tf.concat([data[key] for key in ['train_en', 'train_lv', 'train_ru', 'train_et', 'train_lt']], axis=0)
train_all_labels = tf.concat(
    [data[key] for key in
    ['train_en_labels', 'train_lv_labels', 'train_ru_labels', 'train_et_labels', 'train_lt_labels']], axis=0
)
train_all_validation = tf.concat(
    [data[key] for key in
    ['train_en_validation', 'train_lv_validation', 'train_ru_validation', 'train_et_validation', 'train_lt_validation']
    ], axis=0
)
train_all_labels_validation = tf.concat(
    [data[key] for key in
    ['train_en_labels_validation', 'train_lv_labels_validation',
    'train_ru_labels_validation', 'train_et_labels_validation',
    'train_lt_labels_validation']], axis=0
)

data.update({'train_all': train_all, 'train_all_labels': train_all_labels, 'train_all_validation': train_all_validation,
             'train_all_labels_validation': train_all_labels_validation})

print(data)

classification_all = training(data, "all", learning_rate, sentence_length, batch_size, epochs)

accuracy_en = test_classification_model(classification_all, data, "all_en", batch_size)

accuracy_lv = test_classification_model(classification_all, data, "all_lv", batch_size)

accuracy_ru = test_classification_model(classification_all, data, "all_ru", batch_size)

accuracy_et = test_classification_model(classification_all, data, "all_et", batch_size)

accuracy_lt = test_classification_model(classification_all, data, "all_lt", batch_size)

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


classification_en = training(data, "en", learning_rate, sentence_length, batch_size, epochs)

# ### Test on non-English data


accuracy_lv = test_classification_model(classification_en, data, "lv", batch_size)

accuracy_ru = test_classification_model(classification_en, data, "ru", batch_size)

accuracy_et = test_classification_model(classification_en, data, "et", batch_size)

accuracy_lt = test_classification_model(classification_en, data, "lt", batch_size)

df['5_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

# ### Test on non-English machine translated to English data


accuracy_lv = test_classification_model(classification_en, data, "lv_en", batch_size)

accuracy_ru = test_classification_model(classification_en, data, "ru_en", batch_size)

accuracy_et = test_classification_model(classification_en, data, "et_en", batch_size)

accuracy_lt = test_classification_model(classification_en, data, "lt_en", batch_size)

df['6_method'] = [None, accuracy_lv, accuracy_ru, accuracy_et, accuracy_lt]
print(df)

df.to_csv("many_models.csv")
