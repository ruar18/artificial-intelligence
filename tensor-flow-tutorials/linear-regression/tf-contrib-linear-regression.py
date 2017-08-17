import tensorflow as tf
import numpy as np

# Declare features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# Linear regression estimator
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Training data
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

# Test data, with slightly skewed y's
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# Input helper functions
# The batch size is how much of the data we consider every step
# The number of epochs is the number of iterat
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train},
											   y_train,
											   batch_size=4,
											   num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
	{"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# Train the model on the training set for 1000 iterations
estimator.fit(input_fn=input_fn, steps=1000)

# Evaluate how well the model did
# The train loss should be lower than eval loss, since we specifically fitted
# our model on the training data, after all
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)