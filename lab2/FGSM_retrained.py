# coding: utf-8



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

if(len(sys.argv) < 2):
    print("USAGE: %s <step_size>" % sys.argv[0])
    sys.exit(1)

print("Attacking retrained network witn a step size of %d" % int(sys.argv[1]))
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W1 = tf.Variable(tf.random_normal([784, 300], mean=0, stddev=1))
b1 = tf.Variable(tf.random_normal([300], mean=0, stddev = 1))

#W2 = tf.Variable(tf.random_normal([300, 100], mean=0, stddev= 1))
#b2 = tf.Variable(tf.random_normal([100], mean=0, stddev= 1))

W3 = tf.Variable(tf.zeros([300, 10]))
b3 = tf.Variable(tf.zeros([10]))

# Construct model

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1); #first hidden layer

#hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2); #second hidden layer

pred = tf.nn.softmax(tf.matmul(hidden1, W3) + b3) # Softmax layer outputs prediction probabilities

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
cost_grad = tf.gradients(cost,[x,y])


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


saver = tf.train.Saver()


def non_targeted(x_in, y_in, e):
    y_in = np.roll(y_in,1,axis=1)
    print("USING STEPSIZE ", e)
    x_prime_targeted = (x - (e/256) * tf.sign(cost_grad[0]))/256
    xn = x_prime_targeted.eval({x: x_in,y: y_in})

    print("Accuracy oh new label: ", accuracy.eval({x:xn, y: y_in}))
    print()
    for i in range(xn.shape[0]):
        name = "retrained_FGSM/{}_{}.png".format(str(e),str(i))
        plt.imsave(name,xn[i].reshape(28,28))

with tf.Session() as sess:
    saver.restore(sess, 'remodel.ckpt')
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    label = tf.argmax(pred, 1)[0]
    mean_label=tf.reduce_mean(label)

    non_targeted(mnist.test.images[:3000], mnist.test.labels[:3000],int(sys.argv[1]))

