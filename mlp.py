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

# Display option
display_step = 1      # Display cost after each epoch

# Function for normalization of inputs to a range of 0-1
def norm(x):
    return (x - x.min()) / (x.max() - x.min())

# Load training data
data = pd.read_csv("data/train.csv")
data = data.dropna(how = 'any') # remove NaN

# Store target
y_data = data['target']             # get column
y_data = y_data.astype('category')  # Convert y to numeric
y_data = y_data.cat.codes           #  category codes
y_data = pd.get_dummies(y_data)     # make 1-hot coding

# Get the predictors
x_data = data.ix[:, 1:94] # 93 predictors (0-indexing)
x_data = norm(x_data) # Normalize predictors

# Build training and validation data
idx_train = np.random.rand(len(data)) < 0.7  # 70% training data
x_train = x_data.ix[idx_train]               # Predictor training data
x_test  = x_data.ix[~idx_train]              # Predictor test data
y_train = y_data[idx_train]                  # Target training data
y_test = y_data[~idx_train]                  # Target test data

# Number of rows and cols
n_row = x_data.shape[0]
n_row_train = x_train.shape[0]
n_col_x = x_data.shape[1]
n_col_y = y_data.shape[1]
print(n_row, n_row_train, n_col_x, n_col_y)

# Number of inputs and classes
n_input = n_col_x
n_classes = n_col_y

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Parameters of neural net
pars = {
    'learning_rate': 0.05,  # learning rate of optimizer
    'epochs': 1,            # number of times the training data is presented to the net
    'batch_size': 512       # Number of examples per batch
}
# Neurons in hidden and output layers
neurons = {
    'h1': 512,
    'h2': 256,
    'h3': 128,
    'out': 9
}
# Store layers weight & bias in dictionary
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, neurons['h1']])),
    'h2': tf.Variable(tf.random_normal([neurons['h1'], neurons['h2']])),
    'h3': tf.Variable(tf.random_normal([neurons['h2'], neurons['h3']])),
    'out': tf.Variable(tf.random_normal([neurons['h3'], neurons['out']]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([neurons['h1']])),
    'b2': tf.Variable(tf.random_normal([neurons['h2']])),
    'b3': tf.Variable(tf.random_normal([neurons['h3']])),
    'out': tf.Variable(tf.random_normal([neurons['out']]))
}

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    # Hidden layer 1 with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    # Hidden layer 2 with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    # Hidden layer 3 with RELU activation
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    # Output layer
    return tf.matmul(layer_3, _weights['out']) + _biases['out']

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# Adam optimizer (gradient descent fails somehow)
optimizer = tf.train.AdamOptimizer(pars['learning_rate']).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Model training
with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)
    # Loop epochs
    for epoch in range(pars['epochs']):
        # Reset cost
        avg_cost = 0.
        # Number of batches
        n_batch = int(n_row_train/pars['batch_size'])
        # Loop batches
        for i in range(total_batch):
            # Random sample of training rows
            rows = rd.sample(range(len(x_train)), pars['batch_size'])
            batch_x = x_train.ix[rows, :].dropna()
            batch_y = y_train.ix[rows, :].dropna()
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y}) / n_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%03d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
    # End
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Calculate accuracy of test sample
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
