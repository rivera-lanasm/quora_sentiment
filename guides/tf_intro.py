
# https://www.tensorflow.org/guide/effective_tf2
# eager execution
# functions, not sessions --> tf.function()
# convert python constructs into TF equivalent

# for/while -> tf.while_loop (break and continue are supported)
# if -> tf.cond
# for _ in dataset -> dataset.reduce

# In TensorFlow 2.0, users should refactor their code into smaller functions that are called 
# as needed. In general, it's not necessary to decorate each of these smaller functions with tf.function; 
# only use tf.function to decorate high-level computations - for example, one step of training or the forward 
# pass of your model.

# TODO import tensorflow as tf 
# TODO n = 3
# TODO hidden_size = 2

# Each layer can be called, with a signature equivalent to linear(x)
# TODO layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
# TODO perceptron = tf.keras.Sequential(layers)

# TODO print(layers[2].trainable_variables)
#! perceptron.trainable_variables

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]

## =====================================================

#? https://www.curiousily.com/posts/tensorflow-2-and-keras-quick-start-guide/
import tensorflow as tf 

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

# Tensors
# x = tf.constant(1)
# x

# m = tf.constant([[1, 2, 1], [3, 4, 2]])
# m


# Helpers
# ones = tf.ones([3, 3])
# ones

# zeros = tf.zeros([2, 3])
# zeros

# tf.reshape(zeros, [3, 2])
# tf.transpose(zeros)

# Tensor math
a = tf.constant(1)
b = tf.constant(1)

tf.add(a, b).numpy()

(a + b).numpy()
