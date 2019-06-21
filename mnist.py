import cv2
import keyboard
import random
import tensorflow as tf
import numpy as np
import os
from pynput.mouse import Controller
from PIL import ImageGrab
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(777)  # for reproducibility
from tensorflow.examples.tutorials.mnist import input_data

nb_classes = 10

def screen_capture():
    
    print('press up and down or exit to press ESC')

    while True:
        if keyboard.is_pressed('up'):
            print('up is pressed')
            (x0, y0) = Controller().position
            break
        elif keyboard.is_pressed('esc'):
            exit()

    while True:
        if keyboard.is_pressed('down'):
            print('up is down')
            (x1, y1) = Controller().position
            break
        elif keyboard.is_pressed('esc'):
            exit()
            
    if x0 < x1:
        p0 = x0
        p2 = x1
    elif x0 > x1:
        p0 = x1
        p2 = x0
        
    if y0 < y1:
        p1 = y0
        p3 = y1
    elif y0 > y1:
        p1 = y1
        p3 = y0
        
    printscreen = np.array(ImageGrab.grab(bbox=(p0, p1, p2, p3)))
    dst = cv2.resize(printscreen, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    #cv2.imshow("image",dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    result = []
    
    for m in dst:
        for n in m:
            result.append(max(255 - n) / 255)

    return result


def mnist():
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    # MNIST data image of shape 28 * 28 = 784
    X = tf.placeholder(tf.float32, [None, 784])
    # 0 - 9 digits recognition = 10 classes
    X_img = tf.reshape(X, [-1, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, nb_classes])

    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

    # Final FC 7x7x64 inputs -> 10 outputs
    W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                         initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]))
    logits = tf.matmul(L2_flat, W3) + b

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


    
    # Hypothesis (using softmax)
    
    # Test model
    # Calculate accuracy

    SAVER_DIR = "mnist_cnn"
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, "mnist_cnn")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    
    # parameters
    training_epochs = 30
    batch_size = 100

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            
            while True:
                dst = screen_capture()
                dst = np.ravel(dst,order='C')
                dst = np.array(dst).reshape(1,784)
                saver.restore(sess, ckpt.model_checkpoint_path)    
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
                print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: dst}))

            #sess.close()
            #exit()
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _ = sess.run([cost, optimizer], feed_dict={
                                X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch
            saver.save(sess, 'mnist_cnn/my_model')

            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.9f}'.format(avg_cost))

        print("Learning finished")


        # Test the model using test sets
#        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        dst = np.array(dst).reshape(1,784)
        # Get one and predict
#        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
#        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#        print('Accuracy:', sess.run(accuracy, feed_dict={
#              X: mnist.test.images, Y: mnist.test.labels}))
#        print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: dst}))

#img = screen_capture()

#img = np.array(img).reshape(1,784)

mnist()