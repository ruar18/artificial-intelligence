import numpy as np
import tensorflow as tf

# To define a custom model, need to provide
# tf.contrib.learn.Estimator a few things
def model(features, labels, mode):
	# Build a linear model; Gets existing variables
	W = tf.get_variable("W", [1], dtype=tf.float64)
	b = tf.get_variable("b", [1], dtype=tf.float64)
	# Predictions
	y = W*features['x'] + b

	# Loss subgraph
	loss = tf.reduce_sum(tf.square(y-labels))

	# Training subgraph
	global_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	# tf.group groups multiple operations into a subgraph
	train = tf.group(optimizer.minimize(loss),
					 tf.assign_add(global_step, 1))

	# Connect our subgraphs to their appropriate functionalities
	return tf.contrib.learn.ModelFnOps(
		mode=mode, predictions=y, loss=loss, train_op = train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

# Define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn(
	{"x": x_train}, y_train, 4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
	{"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# Train
estimator.fit(input_fn=input_fn, steps=1000)

# Evaluate the model
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"%train_loss)
print("eval loss: %r"%eval_loss)