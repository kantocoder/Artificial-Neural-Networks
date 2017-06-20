#!/usr/bin/python

import tensorflow as tf
import numpy as np
import copy
import sys

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_output_to_deriv(x):
    return (1-x)*x
    
    
    
# generating training dataset
int2binary={}
binary_dim=8

largest_num=pow(2,binary_dim)

binary=np.unpackbits( np.array([range(largest_num)],dtype=np.uint8).T, axis=1)

for i in range(largest_num):
    int2binary[i]=binary[i]
    
#print np.array([range(largest_num)])
#print int2binary


# input variables
learning_rate = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

# neural network layers
layer_0 = 2*np.random.random((input_dim, hidden_dim))-1
layer_h = 2*np.random.random((hidden_dim,hidden_dim))-1
layer_1 = 2*np.random.random((hidden_dim, output_dim))-1

layer_0_update = np.zeros_like(layer_0)
layer_h_update = np.zeros_like(layer_h)
layer_1_update = np.zeros_like(layer_1)

# print layer_1_update 

for i in range(100000):
    
    a_int = np.random.randint(largest_num/2)
    a = int2binary[a_int]
    
    b_int = np.random.randint(largest_num/2)
    b = int2binary[b_int]

    # expected answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # network predictions storage
    d = np.zeros_like(c)
    
    totalError = 0
    
    layer_2_deltas = list()

    layer_1_values = list()

    layer_1_values.append(np.zeros(hidden_dim))

#    print a
#    print d
    # moving along the positions in the binary encodings
    for position in range(binary_dim):
        #
        X = np.array([[ a[binary_dim - position -1],  b[binary_dim - position -1] ]])
        y = np.array([[ c[binary_dim - position -1] ]]).T
        
#        print position, binary_dim - position -1, X

        # hidden layer
        l1 = sigmoid (np.dot(X, layer_0) + np.dot(layer_1_values[-1], layer_h))
        
        # output layer
        l2 = sigmoid (np.dot(l1, layer_1))
        
        # error computation
        layer_2_error = y - l2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_deriv(l2))
        
        totalError += np.abs(layer_2_error[0])
        
        # decode the estimate
        d[binary_dim - position - 1] = np.round(l2[0][0])
        
        # store hidden layer
        
        layer_1_values.append(copy.deepcopy(l1))

    fut_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[ a[position], b[position] ]])
        l1 = layer_1_values[-position-1]
        prev_l1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]

        # error at hidden layer        
        layer_1_delta = (fut_layer_1_delta.dot(layer_h.T) + layer_2_delta.dot(layer_1.T))*sigmoid_output_to_deriv(l1)


        layer_1_update += np.atleast_2d(l1).T.dot(layer_2_delta)
        layer_h_update += np.atleast_2d(prev_l1).T.dot(layer_1_delta)
        layer_0_update += X.T.dot(layer_1_delta)
        
        fut_layer_1_delta = layer_1_delta

        
    layer_0 += layer_0_update*learning_rate
    layer_h += layer_h_update*learning_rate
    layer_1 += layer_1_update*learning_rate

    layer_0_update = 0
    layer_h_update = 0
    layer_1_update = 0
    
    if (i % 1000 == 0):
        print "Error:"+str(totalError)
        print "Predict: " + str (d)
        print "Expected:" + str (c)
        out = 0
        
        
    
#print layer_1_values
    
sys.exit()


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

