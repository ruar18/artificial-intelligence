from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# Download the datasets if they are not already downloaded
if not os.path.exists(IRIS_TRAINING):
    raw = urrlib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, 'w') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urrlib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, 'w') as f:
        f.write(raw)

# Load datasets
# target_dtype is datatype of labels, features_dtype is datatype of features

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


########## DEFINE MODEL ##########
# Specify that features are real-valued (and that there are 4 of them)
# We can now refer to feature_columns as "x"
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build a 3 layer NN with 10, 20, 30 hidden units
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns
                           hidden_units=[10, 20, 30],
                           n_classes=3,
                           model_dir="/tmp/iris_model")

# Training input pipeline
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

# Train model
classifier.train(input_fn=train_input_fn, steps=2000)
