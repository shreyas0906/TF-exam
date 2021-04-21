import tensorflow as tf

x = tf.constant(8.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    grad = tape.gradient(y, x)

m = tf.constant([1, 2, 3, 4], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x ** 2)
    z = tf.math.sin(y)
    # dz_dy = tape.gradient(z, y)
    dz_dy, dz_dx = tape.gradient(z, [y, x])

x = tf.constant([-1, 0, 1], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.exp(x)
    z = 2 * tf.reduce_sum(y)
    dz_dx = tape.gradient(z, x)




# print(dz_dy)
print(dz_dx)