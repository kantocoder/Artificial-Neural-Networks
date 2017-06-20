import tensorflow as tf

# constructing the graph
x1=tf.constant(5)
x2=tf.constant(6)

#result = x1*x2
result = tf.mul(x1, x2) # abstract tensor in our computation graph, no computation is actually done

print (result)

sess = tf.Session()
r = sess.run(result)
print (r)

sess.close()  # need to close session

# or
with tf.Session() as sess:
	output = sess.run(result)	# output is a python variable
	print(output)

print (output)
