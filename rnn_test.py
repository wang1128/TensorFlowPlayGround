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
    #hidden layer for input to cell
    #X(128 batch, 28 steps, 28 inputs)
    #转化为 (128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # -1 can also be used to infer the shape  https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape
    # X_in ==> (128 batch * 28 steps, 128 hiddern)
    X_in = tf.matmul(X,weights['in']) + biases['in']
    # X_in ==> (128 batch * 28 steps, 128 hiddern)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])


    #cell
    #########
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias= 1.0, state_is_tuple= True) # tuple 数列 (1,2,3)
    # lstm cell is divided into two parts c_state and m_state c_state 主
    _init_state = lstm_cell.zero_state(batch_size= batch_size, dtype= tf.float32)
    #output , states are  list   time major 就是那个step 是不是在第一个维度 这里是第二个维度 所以是false
    outputs, states = tf.nn.dynamic_rnn(cell= lstm_cell, inputs= X_in, initial_state= _init_state, time_major= False)


    # hidden layer for outputs as the final results
    # stat is the last outputs in this project
    results = tf.matmul(states[1],weights['out']) + biases['out']

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1








