#!/usr/bin/python

import tensorflow as tf
import numpy as np


# input variable, tf placeholder
x=tf.placeholder(tf.float32, [None,1])

# actual output value,
y_act=tf.placeholder(tf.float32, [None,1])

# Placeholders for the variables to be trained
W=tf.Variable(tf.zeros([1,1]))      #  W is a 1x1 matrix, or scalar in this case
b=tf.Variable(tf.zeros([1]))            # b is a scalar 
#y=tf.Variable(tf.zeros([1]))

# defining operations
y = tf.matmul(x,W)+b

# cost function
cost_func = tf.reduce_sum(tf.pow((y_act-y),2))

# input data
xs=np.transpose(np.array([np.r_[1:101]]))
ys=2*xs
print xs.shape, ys.shape

# Training
step_size=0.0000001

train_step=tf.train.GradientDescentOptimizer(step_size).minimize(cost_func)

# running the training

sess=tf.Session()
sess.run(tf.initialize_all_variables())

for s in range (1000):
    _, cost_value = sess.run([train_step, cost_func], feed_dict = {x: xs, y_act: ys})
    print ("Iteration %d : W=%f, b=%f, cost=%f"%(s, sess.run(W), sess.run(b), cost_value ))

