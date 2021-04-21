import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
import numpy as np

test_input = [[4,12,33,18],
              [63,23,54,30,19,3],
              [43,91,11,13,15]]

preprocessed_data = pad_sequences(test_input, padding='pre', maxlen=5, truncating='post')
preprocessed_data = preprocessed_data[..., np.newaxis] #(batch_size, seq_length, features)
masking_layer = Masking(mask_value=0)
masked_input = masking_layer(preprocessed_data)

print(preprocessed_data)

print(masked_input)
print(masked_input._keras_mask) # this will tell which inputs must be considered and which should be rejected
