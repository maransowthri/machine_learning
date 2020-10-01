import tensorflow as tf


rank0_string_tensor = tf.Variable('Hello World!', tf.string)
rank0_number_tensor = tf.Variable(324, tf.int16)
rank0_float_tensor = tf.Variable(12.34, tf.float64)
# print(rank0_string_tensor, rank0_number_tensor, rank0_float_tensor)

rank1_string_tensor = tf.Variable(['Hello','World'], tf.string)
# print(rank1_string_tensor)

rank2_string_tensor = tf.Variable([['Hello','World'], ['I"m', 'Maran']], tf.string)
# print(rank2_string_tensor)
# print(tf.rank(rank0_string_tensor))
# print(rank2_string_tensor.shape)

# Reshapes examples

rank2_ones = tf.ones([3, 2, 1], dtype=tf.int16)
rank2_ones = tf.reshape(rank2_ones, [2, 1, 3])
print(rank2_ones)