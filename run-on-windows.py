from model import get_dataset, labels_to_categorical, split_validation, training, convert_to_embeddings

# ## Hyperparameters

batch_size = 25
sentence_length = 20
learning_rate = 0.0003
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

# ## Each language has its own model
# ### Using the original NLU-datasets
# #### Train

classification_model_en = training(data, lang="en", learning_rate=learning_rate, sentence_length=sentence_length,
                                   batch_size=batch_size, epochs=epochs)

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

### Using machine translated non-English datasets
#### Train


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
