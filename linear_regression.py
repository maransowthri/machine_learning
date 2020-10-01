from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt
from six.moves import urllib
from tensorflow.compat.v2 import feature_column as fc

import numpy as np
import pandas as pd
import tensorflow as tf

print("==============================================================================")

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
pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()