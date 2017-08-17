# Get the MNIST dataset, 28x28 pixels each
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

######## INPUT ########
"""
mnist.train.images (training set) is a matrix
Each image is 784 pixels, and there are 55000 images, so
the matrix has a shape of [55000, 784] ([#rows, #columns])
"""

######## LABELS ########
"""
Each image has a one-hot vector, which labels what's
in the image such that labels[n] = 1.
An image of "2" would be labelled
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
"""

######## OUTPUT ########
"""
We want a vector of probabilities for each image, such that
output[n] is the probability that the image is an image
of the digit n. We're gonna use softmax regression to do
this, a method that does what we just described above.
"""

######## METHOD ########
"""
We're gonna somehow (described later) learn weights
for each class (digit). The weights for class i are
denoted W_i (it's a 784-dimensional vector). For each
pixel, the weight that corresponds to the pixel tells us
how the pixel contributes to the likelihood that an image
belongs to the class.
We're then gonna do a weighted sum of all the pixels in an image
(with each W_i), and apply softmax to the weighted sumto get probabilities for each class.
y = softmax(Wx + b)
"""

######## CODE ########
import tensorflow as tf
# Placeholder to store all our images (input)
# We put "none" for the number of images since we want to
# be able to choose the #images later
x = tf.placeholder(tf.float32, [None, 784])
# Placeholder to store our labels
y_ = tf.placeholder(tf.float32, [None, 10])


# Model parameters are variables since they're gonna change
# throughout our computation.
# We have 10 weight vectors (1 for each outcome), and they
# have 784 elements each (one weight for each pixel)
W = tf.Variable(tf.zeros([784, 10]))
# Our bias vector (one bias for each of the 10 classes)
b = tf.Variable(tf.zeros([10]))
# Define our model. Doing xW will result in a matrix with shape
# [#images, 10]. Each row will be the probabilities
# that the image for that row is any of the 10 classes.
# then we add the bias, a 10-element row vector.
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Our cost function, the cross-entropy function
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Train the model using backpropagation and gradient descent
# As you might recall, gradient descent changes our params
# towards the minimum of our cost function (by using the gradient,
# or partial derivatives). Backpropagation is an efficient
# algorithm for computing those partial derivatives.
# train_step does a single gradient descent step when run within a session.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Make a session, which controls the computation of our graph
sess = tf.InteractiveSession()

# Initialize our variables (parameters, like weights and biases)
tf.global_variables_initializer().run()

# Training loop. Repeatedly apply the single gradient descent
# step in order to optimize our cross-entropy cost function.
for _ in range(1000):
	# Get a batch (size: 100 images) of training examples
	# We're only using a small batch at a time since it's
	# computationally cheaper
	batch_xs, batch_ys = mnist.train.next_batch(100)
	# Run the train step on the current batch
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Evaluate the model
# correct_prediction is an array of booleans:
# True where our prediction matches the actual label
# False otherwise
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# To determine what fraction of the images we got right,
# we convert the booleans into doubles (i.e. 1's and 0's),
# then compute the average
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Print the accuracy on the test data
# This should be 90-92%, which is bad
print(sess.run(accuracy*100, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))