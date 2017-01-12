import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001  # learning rate
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST data input (img shape: 28*28) 一行
n_steps = 28    # time steps 28列
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# place holder init
x = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#define Weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), #inputs = 28, hidden neurons = 128 前面定义的
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    #X(128 batch, 28 steps, 28 inputs)
    #转化为 (128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs]) # -1 can also be used to infer the shape  https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape
    



