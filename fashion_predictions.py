from matplotlib import pyplot as plt
from tensorflow import keras

import tensorflow as tf



mnist =  keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# plt.imshow(training_images[0])
# print(training_images[0])
# print(training_labels)

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(training_images, training_labels, epochs=5)
print(model.evaluate(test_images, test_labels))

classifications = model.predict(test_images)
print(classifications[0])