#!/usr/bin/python

# launch TensorBoard as: python /usr/lib/python2.7/site-packages/tensorflow/tensorboard tensorboard.py --logdir ./event_log_dir --port 8888

# Regression with visualization via TensorBoard

import tensorflow as tf
import numpy as np

# input variable, tf placeholder
x=tf.placeholder(tf.float32, [None,1])

# actual output value,
y_act=tf.placeholder(tf.float32, [None,1])

W=tf.Variable(tf.zeros([1,1]))    # W is a 1x1 matrix, or scalar in this case
b=tf.Variable(tf.zeros([1]))      # b is a scalar 
y=tf.Variable(tf.zeros([1]))

# defining operations
y = tf.matmul(x,W)+b

# cost function
cost_func = tf.reduce_sum(tf.square(y_act-y))


# input data
xs=np.transpose(np.array([np.r_[1:101]]))
ys=2*xs

# Training
step_size=0.00000001
train_step=tf.train.GradientDescentOptimizer(step_size).minimize(cost_func)


sess=tf.Session()
sess.run(tf.initialize_all_variables())

# running the training
prev_cost_value = 0
epsilon = 0.0000001
s = 1

while True:
    _, cost_value = sess.run([train_step, cost_func], feed_dict = {x: xs, y_act: ys})
    print ("Iteration %d : W=%f, b=%f, cost=%.10f"%(s, sess.run(W), sess.run(b), cost_value ))
    if abs(cost_value - prev_cost_value)<epsilon:
        print ("Change of cost function is less than %.10f.  Breaking training loop"%epsilon)
        break
    prev_cost_value = cost_value
    s = s+1

