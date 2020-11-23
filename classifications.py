from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt
from six.moves import urllib
from tensorflow.compat.v2 import feature_column as fc

import numpy as np
import os
import pandas as pd
import tensorflow as tf


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

df_train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
df_test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# print(df_train.head())

y_train = df_train.pop('Species')
y_test = df_test.pop('Species')

# print(df_train.head())
# print(y_train.head())
# print('Train data shape', df_train.shape)
# print('Test data shape', df_test.shape)

feature_columns = []

for key in df_train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key, dtype=tf.float32))

# print('Feature Columns', feature_columns)

def input_fn(features, labels, training=True, batch_size=256):
    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        ds = ds.shuffle(1000).repeat()
    
    return ds.batch(batch_size)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[30, 10], n_classes=3)

classifier.train(input_fn=lambda: input_fn(df_train, y_train), steps=5000)
result = classifier.evaluate(input_fn=lambda: input_fn(df_test, y_test, training=False))

# print(result['accuracy'])

# result = list(classifier.predict(input_fn=lambda: input_fn(df_test, y_test, training=False)))
# os.system('CLS')
# print(df_test.loc[0])
# print(y_test.loc[0])
# print(result[0]['probabilities'])

def input_fn_predict(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

for feature in features:
    val = input(feature + ': ')
    predict[feature] = [float(val)]

predictions = list(classifier.predict(input_fn=lambda: input_fn_predict(predict)))

pred_dict = predictions[0]
class_id = pred_dict['class_ids'][0]
probability = pred_dict['probabilities'][class_id]

os.system('CLS')
print(f'Prediction is {SPECIES[class_id]} ({probability*100:.3f})')