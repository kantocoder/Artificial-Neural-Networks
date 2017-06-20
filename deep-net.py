'''
SUMMARY:

input > weight > hidden layer 1 (activation function) > weight >hidden l 2 (activation function) > weight > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer ... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch

'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)  # one_hot 

'''  under one_hot = True, one element is hot, the rest are zero (cold)
 10 classes 0..9

0 = [1,0,0,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0,0,0]

'''

n_nodes_hl1=500 # number of nodes in hidden layer 1
n_nodes_hl2=500 # number of nodes in hidden layer 2
n_nodes_hl3=500 # number of nodes in hidden layer 3

n_classes = 10   # number of classes
batch_size = 100  #  batches of 100 features are feed at a time and manipulate the weights

# heigh x width 784
x = tf.placeholder('float',[None, 784]) #input data  
y = tf.placeholder('float')

def neural_network_model(data):

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])), 
					   'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])), 
			 		   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])), 
				 	   'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer =   {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])), 
				       'biases': tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weihgts) + biases
	l1 = tf.add( hidden_1_layer['biases'] , tf.matmul(data, hidden_1_layer['weights']) )
	l1 = tf.nn.relu(l1)  #rectified linear, activation function

	l2 = tf.add( hidden_2_layer['biases'] , tf.matmul(l1, hidden_2_layer['weights']) )
	l2 = tf.nn.relu(l2)  #rectified linear, activation function

	l3 = tf.add( hidden_3_layer['biases'] , tf.matmul(l2, hidden_3_layer['weights']) )
	l3 = tf.nn.relu(l3)  #rectified linear, activation function

	output = output_layer['biases'] + tf.matmul(l3, output_layer['weights']) 
	return (output)
	
def	train_neural_network(x):
	prediction=neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y) )

	#  learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + backprop
	hm_epochs = 10	# how many epochs

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		
'''		# training the network
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_lost +=c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
'''

train_neural_network(x)
