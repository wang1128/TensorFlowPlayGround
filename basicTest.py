#This is the first basic test of tensorflow

import tensorflow as tf
import numpy as np

def basic_operation():
    v1 = tf.Variable(10)
    v2 = tf.Variable(5)
    addv = v1 + v2

    sess = tf.Session()

    tf.initialize_all_variables().run(session = sess)

    print(addv.eval(session = sess))
    print(sess.run(addv))

    graph = tf.Graph()
    with graph.as_default():
        value1 = tf.constant([1,2])
        value2 = tf.constant([3,4])
        mul = value1 * value2  #不是线代的乘法 #不是内积 一一对整的乘

    with tf.Session(graph=graph) as mySess:
        tf.initialize_all_variables().run()
        print(mySess.run(mul))

    #placeholder


def use_placeholder():
    return


basic_operation()

