import tensorflow as tf
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))


def rescale(image, label):
    return image / 255., label


def label_filter(image, label):
    return tf.squeeze(label) != 9 # this means any sample in the data whose label is not 9  will be filtered out


dataset = dataset.map(rescale)
dataset = dataset.filter(label_filter)

dataset = dataset.shuffle(100)
dataset = dataset.batch(16,
                        drop_remainder=True)  # if the entine_datset // batch_size != 0, the drop_remainder will either include it or ignore it.
dataset = dataset.repeat()  # this can be used as epochs or repeat the dataset indefinitely.

history = model.fit(dataset)
