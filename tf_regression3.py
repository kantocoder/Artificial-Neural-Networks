#!/usr/bin/python

# launch TensorBoard as: python /usr/lib/python2.7/site-packages/tensorflow/tensorboard tensorboard.py --logdir ./event_log_dir --port 8888

# Regression with visualization via TensorBoard

import tensorflow as tf
import numpy as np


# input variable, tf placeholder
x=tf.placeholder(tf.float32, [None,1], name = 'x')

# actual output value,
y_act=tf.placeholder(tf.float32, [None,1], name = 'y_act')

# Placeholders for the variables to be trained
with tf.name_scope('Variables') as scope:
  W=tf.Variable(tf.zeros([1,1]), name='W')     # W is a 1x1 matrix, or scalar in this case
  b=tf.Variable(tf.zeros([1]),   name='b')     # b is a scalar 
  y=tf.Variable(tf.zeros([1]),   name='y')


# W and b can be multi-dimensional so 
# mark them as data to collected using histogram_summary
W_hist=tf.histogram_summary("weights", W)
b_hist=tf.histogram_summary("biases", b)


# defining operations
with tf.name_scope ("Wx_b") as scope:
  y = tf.matmul(x,W)+b

# cost function
with tf.name_scope ("cost_function") as scope:
  cost_func = tf.reduce_sum(tf.square(y_act-y))


# cost_func is a 1-dim, so mark it as a data to be
# collected using scalar_summary
cost_sum = tf.scalar_summary("cost_function", cost_func)



# input data
xs=np.transpose(np.array([np.r_[1:101]]))
ys=2*xs
print xs.shape, ys.shape

# Training
step_size=0.00000001
with tf.name_scope ("train_step") as scope:
  train_step=tf.train.GradientDescentOptimizer(step_size).minimize(cost_func)


# TensorFlow Summary: Merge
merged = tf.merge_all_summaries()

#merged = tf.merge_summary([W_hist, b_hist, cost_sum])


sess=tf.Session()

# Tensorflow Writer
# The specified log directory will be:
#  + Create if it does not exist
#  + If it exists, and not empty the data are merged
# sess.graph_def will enable Tensorflow to draw graph 
# representation for the network
writer=tf.train.SummaryWriter("./event_log_dir", sess.graph_def)


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
    
    if s % 10 == 0:
      result = sess.run(merged, feed_dict={x: xs, y_act: ys})
      writer.add_summary(result, s)
