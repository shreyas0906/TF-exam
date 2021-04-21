from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import SGD
import tensorflow as tf


# This is model subclassing.
class myModel(Model):
    """
    The layers of the custom model will be initialized in the init method
    """

    # def __init__(self, num_classes, **kwargs):
    def __init__(self, hidden_units, outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = Dense(16, activation='sigmoid')
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(hidden_units, activation='softmax')
        self.linear = LinearMap(hidden_units, outputs) # the number of inputs to the LinearMap layer must match the number of outputs of the previous layer.

    """
    This is to call the forawrd pass using the layers in the model which we have defined in __init__ method
    training determines the behaviour of the model at train time and test time. Used mainly in dropout and batch_norm layers. 
    """

    def call(self, inputs, training=False):
        h = self.dense1(inputs)
        # h = self.dropout(h)
        # return self.dense2(h)
        return self.linear(h)


# This is layer subclassing
class LinearMap(Layer):
    """
    Instead of creating layers in the initializer, we are creating the layer variables in the initializer.
    The call method contains the layer computation.
    See the ipython notebook for more details.
    """

    def __init__(self, input_dim, units):
        super().__init__()
        w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units)))
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal')
        print(f"self.w.shape: {self.w.shape}")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


linear_layer = LinearMap(3,2)
inputs = tf.ones((1,3))
my_model = myModel()

optimizer = SGD(learning_rate=0.05, momentum=0.9)

def loss(y_hat, y):
    return tf.reduce_mean(tf.square(y_hat - y))

with tf.GradientTape() as tape:

    current_loss = loss(my_model(inputs), outputs) # This is a placeholder for a custom loss function, it can be replaced by using in-built loss functions.
    grads = tape.gradient(current_loss, my_model.trainable_variables) # This will calculate the gradients of the loss w.r.t model parameters.


optimizer.apply_gradients(zip(grads, my_model.trainable_variables)) # this is to update the weights of the model with the new gradients


# print(inputs)
# print(linear_layer(inputs))
# print(linear_layer.weights)
my_model = myModel(10, name='my_model')