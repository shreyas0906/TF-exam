import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD

"""
1. Define custom model and custom layer 
2. take a training dataset, train on model with custom training loops. 
"""

class MyModel():
    ...


my_model = MyModel()
loss = MeanSquaredError()
optimizer = SGD(learning_rate=0.05, momentum=0.9)

batch_loss = []

for inputs, outputs in training_dataset:
    with tf.GradientTape() as tape:
        current_loss = loss(my_model(inputs), outputs)
        grads = tape.gradient(current_loss, my_model.trainable_variables)

    batch_loss.append(current_loss)
    optimizer.apply_gradients(zip(grads, my_model.trainable_variables))

