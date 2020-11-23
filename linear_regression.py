from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt
from six.moves import urllib
from tensorflow.compat.v2 import feature_column as fc

import numpy as np
import os
import pandas as pd
import tensorflow as tf


df_train = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
df_eval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
# print(df_train.head())
# print(df_train.describe())
# print(df_train.shape)
# print(df_train.ndim)

y_train = df_train.pop('survived')
y_eval = df_eval.pop('survived')
# print(df_train.head())
# print(y_train.head())
# print(df_train.iloc[0], y_train.iloc[0])
# df_train.age.hist(bins=20)
# df_train.sex.value_counts().plot(kind='barh')
# pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()

CATAGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

# print(df_train["sex"].unique())

for feature_name in CATAGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(df_train, y_train)
eval_input_fn = make_input_fn(df_eval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

# print('Accuracy', result['accuracy'])
# print(result)

result = list(linear_est.predict(eval_input_fn))
os.system('CLS')
print(df_eval.loc[2])
print(y_eval.loc[2])
print(result[2]['probabilities'])