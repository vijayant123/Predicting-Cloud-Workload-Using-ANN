import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Reading dataset


def ret_mse():
    return mse_history


def read_data():
    df = pd.read_csv("eggs_new.csv")
    X = df[df.columns[4:6]].values
    y = df[df.columns[11]]

    # Encoding the dependant variable

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    return X, Y


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Read the dataset

X, Y = read_data()

# Shuffle the rows

X, Y = shuffle(X, Y, random_state=1)

# Convert the dataset in training and testing dataset

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.01, random_state=415)

# defining important params and variables to work with Tensors

learning_rate = 0.001                                # Step size
n_epochs = 200                                         # epoch = forward + backward propagation
cost_history = np.empty(shape=[1], dtype=float)        # A numpy array to facilitate in prediction using history
n_dim = X.shape[1]                                     # Dimension for Tensor = 1D array of inputs
n_classes = 2                                          # Either New VM or Not
model_path = "/home/sagar/Desktop/Pycharm Projects/Minor Project/ANNTensor"

# Defining Layers for Neural Network

nodes_hl1 = 20
nodes_hl2 = 20
nodes_hl3 = 20
nodes_hl4 = 20

input_data = tf.placeholder(tf.float32, [None, n_dim])  # Input to the nodes
W = tf.Variable(tf.zeros(n_dim, n_classes))             # Weights, initially 0
b = tf.Variable(tf.zeros(n_classes))                    # Biases, initially 0
output = tf.placeholder(tf.float32, [None, n_classes])  # output from the neural network


def model_neural_network(x, weigths, biases):

    # Hidden Layer with Sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weigths['h1']), biases['h1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden Layer with Sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weigths['h2']), biases['h2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden Layer with Sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weigths['h3']), biases['h3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden Layer with Sigmoid activation
    layer_4 = tf.add(tf.matmul(layer_3, weigths['h4']), biases['h4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    # Output Layer with RELU activation
    output_layer = tf.matmul(layer_4, weigths['out']) + biases['out']
    output_layer = tf.nn.relu(output_layer)

    return output_layer


# Defining weigths and biases

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, nodes_hl1])),
    'h2': tf.Variable(tf.truncated_normal([nodes_hl1, nodes_hl2])),
    'h3': tf.Variable(tf.truncated_normal([nodes_hl2, nodes_hl3])),
    'h4': tf.Variable(tf.truncated_normal([nodes_hl3, nodes_hl4])),
    'out': tf.Variable(tf.truncated_normal([nodes_hl4, n_classes]))
}

biases = {
    'h1': tf.Variable(tf.truncated_normal([nodes_hl1])),
    'h2': tf.Variable(tf.truncated_normal([nodes_hl2])),
    'h3': tf.Variable(tf.truncated_normal([nodes_hl3])),
    'h4': tf.Variable(tf.truncated_normal([nodes_hl4])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}



# Initialize all global variables

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Call to the model

y = model_neural_network(input_data, weights, biases)

# Cost function and Gradient Descent Optimizer

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=output))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()

sess.run(init)

# Calculate cost fn and accuracy for each epoch

mse_history = []
accuracy_history = []

for epoch in range(n_epochs):
    sess.run(optimizer, feed_dict={input_data: train_x, output: train_y})
    cost = sess.run(cost_function, feed_dict={input_data: train_x, output: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predict_y = sess.run(y, feed_dict={input_data: test_x})
    mse = tf.reduce_mean(tf.square(predict_y - test_y))
    mse_output = sess.run(mse)
    mse_history.append(mse_output)
    accuracy = (sess.run(accuracy, feed_dict={input_data: train_x, output: train_y}))
    accuracy_history.append(accuracy)

    #print ('epoch: ', epoch, ' - cost: ', cost, ' - mse: ', mse_output, " - Train Accuracy: ", accuracy)

save_path = saver.save(sess, model_path)
#print "Model Saved in path: " + save_path

# Plot MSE and Accuracy graph

plt.plot(mse_history, 'r')
plt.xlabel('Iterations')
plt.ylabel('Mean Square Error')
plt.show()
plt.plot(accuracy_history)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.show()

# Print the final accuracy

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={input_data: test_x, output: test_y})))

# Print the final mean square error

predict_y = sess.run(y, feed_dict={input_data: test_x})
mse = tf.reduce_mean(tf.square(predict_y - test_y))
print("MSE: %.4f" % sess.run(mse))




