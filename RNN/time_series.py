import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow.keras as keras


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)

    return series[..., np.newaxis].astype(np.float32)

n_steps = 50 
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, : n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, n_steps], series[9000:, -1]

# plt.plot(X_train[1,:,:], range(0,50))
# plt.show()

y_pred = X_valid[:,-1]
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

# model = keras.models.Sequential([keras.layers.Flatten(input_shape=[50,1]),
#                                 keras.layers.Dense(1)])
#
# model.compile(loss='MSE', optimizer='adam')
# history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),epochs=20)

model = keras.models.Sequential([keras.layers.SimpleRNN(20, input_shape=[None, 1], return_sequences=True),
                                 keras.layers.SimpleRNN(20, return_sequences=True),
                                 keras.layers.SimpleRNN(1)])

print(model.summary())


