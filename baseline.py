# Hao WU, Baseline Implementation
'''
The baseline is basically a Linear Regression Model:
y = W*x + b

where:
x is a vetor representation of English Instructions dimension 16
y is a vetor representation of Unix Commands, dimension 12
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils

# Just gonna share some helper functions
import translate

# train for 1780 steps
training_epochs = 100
learning_rate = math.exp(-11)
display_step = 20

# Use something smaller to train faster
seq_len = 40
# get data
from_train = None
to_train = None
from_dev = None
to_dev = None

from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        'data',
        'data/data.txt',
        'data/label.txt',
        'data/validation/data.txt',
        'data/validation/label.txt',
        40000,
        4000)

# Start training
with tf.Session() as sess:
	# tf Graph Input, assume instruction/command length is maximally 50
	X = tf.placeholder(tf.float32, shape=(1, seq_len))
	Y = tf.placeholder(tf.float32, shape=(1, seq_len))

	# Set model weights
	W = tf.Variable(tf.random_normal([seq_len, seq_len], stddev=0.1), name="weights")
	b = tf.Variable(tf.random_normal([1, seq_len], stddev=0.1), name="bias")

	# Construct a linear model
	pred = tf.add(tf.matmul(X, W), b)

	# Cost is how many values in a sequence mismatch
	cost = tf.log(tf.reduce_sum(tf.pow(pred-Y, 2)))

	#check number of exact matches
	num_mismatch = tf.count_nonzero(tf.cast(pred, dtype=tf.int64)-tf.cast(Y, dtype=tf.int64))
	
	# Gradient descent
	#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	train_set = translate.read_data(from_train, to_train)[0]
	#print(train_set[0])
	dev_set = translate.read_data(from_dev, to_dev)[0]
	# Run the initializer
	sess.run(init)
    # Do stochastic gradient decent
	for epoch in range(training_epochs + 1):
		for (x, y) in train_set:
			#print(x)
			#print(y)
			xx = np.pad(x, (0, seq_len - len(x)%seq_len), 'constant')
			yy = np.pad(y, (0, seq_len - len(y)%seq_len), 'constant')
			x = np.array(xx).reshape(1, seq_len)
			y = np.array(yy).reshape(1, seq_len)
			# reshape the data so it fit the placeholders
			sess.run(optimizer, feed_dict={X: x, Y: y})

		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
			total_cost = 0.0
			for (x, y) in train_set:
				xx = np.pad(x, (0, seq_len - len(x)%seq_len), 'constant')
				yy = np.pad(y, (0, seq_len - len(y)%seq_len), 'constant')
				x = np.array(xx).reshape(1, seq_len)
				y = np.array(yy).reshape(1, seq_len)
				total_cost += math.exp(sess.run(cost, feed_dict={X: x, Y: y}))
			print("Epoch:", '%04d' % (epoch+1), "cost= {:.9f}".format(total_cost/5030.0))

	count_complete_match = 0
	count_match_ratio = []
	count_right_order = 0
	for (x, y) in train_set:
		target_length = len(y)
		xx = np.pad(x, (0, seq_len - len(x)%seq_len), 'constant')
		yy = np.pad(y, (0, seq_len - len(y)%seq_len), 'constant')
		x = np.array(xx).reshape(1, seq_len)
		y = np.array(yy).reshape(1, seq_len)
		n_m = sess.run(num_mismatch, feed_dict={X: x, Y: y})
		pp = sess.run(pred, feed_dict={X: x, Y: y})
		# if there is no mismatch, a complete match
		if n_m == 0:
			count_complete_match += 1

		count_match_ratio.append(max(target_length - n_m, 0)*1.0/target_length)

		if translate.check_order(y, pp):
			count_right_order += 1

	set_name = "Training Set"
	size = 5030
	print("Overall parts matched per command in precentage: {} in {}".format(np.average(count_match_ratio), set_name))
	print("precentage of correct ordered command: {} in {}".format(count_right_order*1.0/size, set_name))
	print("Number of complete matches: {} out of {} in {}".format(count_complete_match, size, set_name))

	print("Optimization Finished!")
	


