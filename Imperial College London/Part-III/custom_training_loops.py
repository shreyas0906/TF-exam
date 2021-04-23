import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import SGD

# my_model = MyModel()
# loss = MeanSquaredError()
# optimizer = SGD(learning_rate=0.05, momentum=0.9)

# with tf.GradientTape() as tape:
#     current_loss = loss(my_model(inputs), outputs) # outputs should be ground truth.
#     grads = tape.gradient(current_loss, my_model.trainable_variables) # calculate the gradients of the model wrt the loss.

#optimizer.apply_gradients(zip(grads, my_model.trainable_variables)) # this is to update the weights.
# this will update the weights of the model using the calculated gradients.



# For a batch of inputs and outputs
# batch_losses = []
# for inputs, outputs in training_dataset:
#     with tf.GradientTape() as tape:
#         current_loss = loss(my_model(inputs), outputs) # forward pass
#         grads = tape.gradient(current_loss, my_model.trainable_variables) # backward pass
#
#     batch_losses.append(current_loss)
#     optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
#


# adding tf.function to make the computation faster as the computation graph is created for peak performance
# my_model = MyModel()
# loss = MeanSquaredError()
# optimizer = SGD(learning_rate=0.05, momentum=0.9)

@tf.function
def get_loss_and_grads(inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(my_model(inputs), outputs)
        grads = tape.gradient(current_loss, my_model.trainable_variables)
    return current_loss, grads

for epoch in range(num_epochs): # loops over the epochs
    for inputs, outputs in training_dataset: # loops over the batches in the training_dataset
        current_loss, grads = get_loss_and_grads(inputs, outputs)
        optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
