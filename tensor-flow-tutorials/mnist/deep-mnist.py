# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create session
import tensorflow as tf

# Placeholder to store our images
x = tf.placeholder(tf.float32, [None, 784])
# Placeholder to store our labels
y_ = tf.placeholder(tf.float32, [None, 10])

# Create weights
def weight_variable(shape):
    # Create random values, normal distribution (for symmetry breaking)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Create biases
def bias_variable(shape):
    # Have a slightly positive bias to avoid dead ReLU
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution layer
def conv2d(x, W):
    # We're moving our filter one row/column at the time
    # Padding will ensure that the output size is the same as input size
    # Side note: convolution is probably O(n log n) (optimized using FFt)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling layer: pick the max neuron from every 2x2 block
# This helps computation efficiency, and controls overfitting
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1,2,2,1], padding='SAME')



######## FIRST CONVOLUTIONAL LAYER ########
# Define our first convolutional layer (convolution followed by
# max pooling). We have 32 5x5 filters, so we'll get 32 numbers (features)
# for every 5x5 patch of our image
# [width, height, input channels, output channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# To apply layer, reshape x into a 4d tensor
# First num is a placeholder (to make the model independent of the #images),
# second and third are width and height, last is #colour channels
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Apply the layer (and ReLU to introduce non-linearity)
# The result is a 28x28x32 tensor of values
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Downsample the image into a 14x14x32 tensor using maxpooling
h_pool1 = max_pool_2x2(h_conv1)


######## SECOND CONVOLUTIONAL LAYER ########
# Now, we make another 64 5x5 filters
# We'll get 64 features for every 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Convolve and ReLU to introduce non-linearity
# Our input is the previous layer's output
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Downsample again: we'll get a 7x7x64 image
h_pool2 = max_pool_2x2(h_conv2)


######## DENSLY/FULLY CONNECTED LAYER ########
# A fully connected layer with 1024 neurons
# Our weight has 7*7*64 inputs, 1024 outputs
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1  = bias_variable([1024])

# We flatten out each of our images into a 7*7*64 image
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# Apply ReLU
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


######## DROPOUT ########
# Dropout is a technique which reduces overfitting by killing
# neurons at random. We only want it on during training (since overfitting
# happens during training), so we'll make a placeholder that allows us
# to turn it off during testing (it represents the probability that
# a neuron isn't killed)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


######## READOUT - FINAL (SOFTMAX) LAYER ########
# Our final layer has 10 neurons, each with the probability
# that the image is that number represented by the neuron.
# We've come a long way, from 28x28, all the way to 10 neurons.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


###################### TRAINING ######################
# Our cost function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# Optimizer (this time an Adam gradient descent algorithm)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Prediction comparison
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# We're putting the session in a "with" block so it's automatically
# destroyed after the block (we don't need it anymore)
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Train for 20k iterations
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        # Report training accuracy every 100 iterations
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})*100
            print('step %d, training accuracy %g%%' % (i, train_accuracy))
        # Perform a training step
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # Report test accuracy after the training is done
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100)
