import tensorflow as tf
import sys
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.datasets import cifar10
import numpy as np

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

# Pre-processing the data set
xTrain = xTrain.astype(np.float)
yTrain = np.squeeze(yTrain)

yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Set hyper parameter
lr = 1e-6
model_path = os.path.join('model', 'cfar10')

# Normalize data by subtracting mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xTest -= meanImage

# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

# Create the model
x = tf.placeholder(tf.float32, [None, 3072])
y_expected = tf.placeholder(tf.int64, [None])

# Variables
W = tf.Variable(tf.zeros([3072, 10]))
b = tf.Variable(tf.zeros([10]))

# Output
y = tf.matmul(x, W) + b

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_expected, 10), logits=y))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), y_expected), tf.float32))

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)


def train():
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    save_model = tf.train.Saver()

    print("Loop\t\tTrain Loss\t\tTrain Acc %\t\tTest loss\t\tTest Acc %")

    # Train
    for i in range(2000):
        s = np.arange(xTrain.shape[0])
        np.random.shuffle(s)
        xTr = xTrain[s]
        yTr = yTrain[s]
        batch_xs = xTr[:128]
        batch_ys = yTr[:128]

        train_loss, train_acc, _ = sess.run([cross_entropy, accuracy, train_step],
                                            feed_dict={x: batch_xs, y_expected: batch_ys})

        test_loss, test_acc = sess.run([cross_entropy, accuracy], feed_dict={x: xTest, y_expected: yTest})

        if i % 200 == 0:
            print('{0}/10\t\t{1:0.6f}\t\t{2:0.6f}\t\t{3:0.6f}\t\t{4:0.6f}'.format(int(i / 200) + 1, train_loss,
                                                                                  train_acc * 100, test_loss,
                                                                                  test_acc * 100))
    save_path = save_model.save(sess, model_path)
    print("Model saved in file: ", save_path)
    sess.close()


labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def test(image_path):
    input_image = cv2.imread(image_path)
    test_image = np.reshape(input_image, (1, -1))

    if np.size(test_image) != 3072:
        print("Image size is not 32 x 32 x 3")
        return
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    save_model = tf.train.Saver()
    save_model.restore(sess, model_path)
    test_loss = sess.run([y], feed_dict={x: test_image})
    print(labels[np.argmax(test_loss)])
    sess.close()

if sys.argv[1] == "train":
    train()
elif sys.argv[1] == "test" or sys.argv[1] == "predict":
    test(sys.argv[2])
else:
    print("Invalid Syntax")