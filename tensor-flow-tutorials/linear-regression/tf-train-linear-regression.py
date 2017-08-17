import tensorflow as tf

# Model params
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input/output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# Loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Session details
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
for i in range(1000):
	sess.run(train, {x: x_train, y: y_train})

final_W, final_b, final_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (final_W, final_b, final_loss))