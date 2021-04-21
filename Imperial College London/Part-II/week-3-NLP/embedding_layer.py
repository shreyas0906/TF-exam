from tensorflow.keras.layers import Embedding
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU

embedding_layer = Embedding(1000, 32, mask_zero=True) # input_length=64
test_input = np.random.randint(1000, size=(16,64))
embedded_inputs = embedding_layer(test_input)
print(embedded_inputs._keras_mask)

model = Sequential([Embedding(1000, 32, input_length=64),
                    SimpleRNN(64, activation='tanh'), # they can take variable length sequences. LSTM(64, activation='tanh')
                    Dense(5, activation='softmax')])

