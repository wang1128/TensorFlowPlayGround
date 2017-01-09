import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


def add_layers(inputs, in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# define placeholder for inputs to network

xs = tf.placeholder(tf.float32,[None,784]) # 28 * 28
ys = tf.placeholder(tf.float32, [None,10])

#add output layer
predictions = add_layers(xs,784,10,activation_function = tf.nn.softmax) #in_size = 784, out_size is result
