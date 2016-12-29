import tensorflow as tf

def load_from_remote():
    return [ x for x in range(10000)]


#yield generator function
def load_partial(value, step):
    index = 0
    while index < len(value):
        yield value[index: index + step]
        index += step
    return

def use_placeholder():
    graph = tf.Graph()
    with graph.as_default():
        value1 = tf.placeholder(dtype=tf.float64)
        value2 = tf.Variable([1,1],dtype=tf.float64)
        mul = value1 * value2

    with tf.Session(graph=graph) as mySess:
        tf.initialize_all_variables().run()

        value = load_from_remote()
        for partialValue in load_partial(value,2):
            runResult = mySess.run(mul, feed_dict={value1:partialValue})
            print(runResult)

use_placeholder()