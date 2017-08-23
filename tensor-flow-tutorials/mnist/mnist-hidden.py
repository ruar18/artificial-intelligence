# My modification of the simple one-layer neural net
# A network with a single hidden layer should improve performance
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
FLAGS = None
import tensorflow as tf

# Get the MNIST dataset, 28x28 pixels each
from tensorflow.examples.tutorials.mnist import input_data

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

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

    # Placeholder to store all our images (input)
    # We put "none" for the number of images since we want to
    # be able to choose the #images later
    x = tf.placeholder(tf.float32, [None, 784])
    # Placeholder to store our labels
    y_ = tf.placeholder(tf.float32, [None, 10])

    n1_hidden = 1000
    # Model parameters are variables since they're gonna change
    # throughout our computation.
    # We have 10 weight vectors (1 for each outcome), and they
    # have 784 elements each (one weight for each pixel)
    W_1 = tf.Variable(tf.truncated_normal([784, n1_hidden], stddev=1.0))
    # Our bias vector (one bias for each of the 10 classes)
    b_1 = tf.Variable(tf.zeros([n1_hidden]))
    # Define our model. Doing xW will result in a matrix with shape
    # [#images, 10]. Each row will be the probabilities
    # that the image for that row is any of the 10 classes.
    # then we add the bias, a 10-element row vector.
    h_1 = tf.nn.relu((tf.matmul(x, W_1) + b_1))

    n2_hidden = 100
    W_2 = tf.Variable(tf.truncated_normal([n1_hidden, n2_hidden], stddev=1.0/math.sqrt(float(n1_hidden))))
    b_2 = tf.Variable(tf.zeros([n2_hidden]))
    h_2 = tf.nn.relu((tf.matmul(h_1, W_2) + b_2))

    W_3 = tf.Variable(tf.zeros([n2_hidden, 10]))
    b_3 = tf.Variable(tf.zeros([10]))
    y = tf.matmul(h_2, W_3) + b_3;
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

    # Evaluate the model
    # correct_prediction is an array of booleans:
    # True where our prediction matches the actual label
    # False otherwise
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # To determine what fraction of the images we got right,
    # we convert the booleans into doubles (i.e. 1's and 0's),
    # then compute the average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Training loop. Repeatedly apply the single gradient descent
    # step in order to optimize our cross-entropy cost function.
    for _ in range(1000):
        # Get a batch (size: 100 images) of training examples
        # We're only using a small batch at a time since it's
        # computationally cheaper
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # Run the train step on the current batch
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # print(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})*100)




    # Print the accuracy on the test data
    # This should be 90-92%, which is bad
    print(sess.run(accuracy*100, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='C:/Users/rudya/Documents/GitHub/learning-machine-learning/tensor-flow-tutorials/mnist/MNIST_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
