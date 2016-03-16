'''
Python tutorial on neural networks with TensorFlow
-------------------------------------------------------
A multilayer neural network based on OTTO Kaggle data
Implementation in Google TensorFlow
'''

# Imports
import pandas as pd
import random as rd
import numpy as np
import tensorflow as tf

# Parameters of neural net
learning_rate = 0.05
epochs = 1
batch_size = 512
display_step = 1

# Function for normalization of inputs
def norm(x):
    return (x - x.min()) / (x.max() - x.min())

# Load training data
data = pd.read_csv("data/train.csv")

# Remove NaN rows
data = data.dropna(how = 'any')

# Take a sample for speed
data = data.sample(20000)
print(data.shape)

# Show first 10 rows
data.head(10)

# Store target
y_data = data['target']

# Convert y to numeric
y_data = y_data.astype('category')
y_data = y_data.cat.codes
y_data = pd.get_dummies(y_data)
y_data.head()

# Get the predictors
x_data = data.ix[:, 1:94]

# Normalize x data
x_data = norm(x_data)

# Build training and validation data
idx_train = np.random.rand(len(data)) < 0.7
x_train = x_data.ix[idx_train]
x_test  = x_data.ix[~idx_train]
y_train = y_data[idx_train]
y_test = y_data[~idx_train]

# Number of rows and cols
n_row = x_data.shape[0]
n_row_train = x_train.shape[0]
n_col_x = x_data.shape[1]
n_col_y = y_data.shape[1]
print(n_row, n_row_train, n_col_x, n_col_y)

# Neurons in hidden layers
neurons = {
    'h1': 512,
    'h2': 256,
    'h3': 128
}
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_hidden_3 = 128 # 2nd layer num features
n_input = n_col_x # MNIST data input (img shape: 28*28)
n_classes = n_col_y # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias in dictionary
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    # Hidden layer with RELU activation
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    return tf.matmul(layer_3, _weights['out']) + _biases['out']

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Model training
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)
    # Loop epochs
    for epoch in range(epochs):
        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(n_row_train/batch_size)
        # Loop batches
        for i in range(total_batch):
            # Random sample of training rows
            rows = rd.sample(range(len(x_train)), batch_size)
            batch_x = x_train.ix[rows, :].dropna()
            batch_y = y_train.ix[rows, :].dropna()
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    # End
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
