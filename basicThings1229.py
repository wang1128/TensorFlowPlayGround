# Mofan youtube 1229

import tensorflow as tf
import numpy as np

# create data float32 tf是常用的
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# create tensorflow structure begin #

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #随机 一维结构 -1 to 1
biases = tf.Variable(tf.zeros([1]))

y= Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) # loss function
optimizer = tf.train.GradientDescentOptimizer(0.5) #0.5 是学习效率
train = optimizer.minimize(loss)

init = tf.initialize_all_variables() #初始

# create tensorflow structure end #

sess = tf.Session()
sess.run(init)

print(x_data)
for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
