import cv2
import time
import numpy as np
import keyboard
import glob
import csv
import tensorflow as tf
import os
from pynput.mouse import Controller
from PIL import ImageGrab

file=open("playdata.csv","r",newline='')#파일을 생성 혹은 오픈
csv_write=csv.writer(file)#연 파일을 csv로 바꿔줌

def Learning():
    learn_variable=1
    nb_classes=2
    data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈
    x_data=data[:,0:-1]
    y_data=data[:, learn_variable:]
    X = tf.placeholder(tf.float32, [None, learn_variable])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, 1])
    
    # dropout_rate=tf.placeholder("float")
    W1 = tf.Variable(tf.random_normal([learn_variable, nb_classes]))
    b1 = tf.Variable(tf.random_normal([nb_classes]))
    logits=tf.nn.relu(tf.matmul(X,W1)+b1)
    
    # W2 = tf.Variable(tf.random_normal([20, 20]))
    # b2 = tf.Variable(tf.random_normal([20]))
    # layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)
    # layer2=(_layer2,dropout_rate)

    # W3 = tf.Variable(tf.random_normal([20, 20]))
    # b3 = tf.Variable(tf.random_normal([20]))
    # layer3=tf.nn.relu(tf.matmul(layer2,W3)+b3)

    # W3 = tf.Variable(tf.random_normal([20, 20]))
    # b3 = tf.Variable(tf.random_normal([20]))
    # layer3=tf.nn.relu(tf.matmul(layer2,W3)+b3)

    # W4 = tf.Variable(tf.random_normal([20,nb_classes]))
    # b4 = tf.Variable(tf.random_normal([nb_classes]))
    # logits = tf.matmul(layer3, W4) + b4

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

    # rate = tf.Variable(0.1)
    # optimizer = tf.train.GradientDescentOptimizer(rate)

    saver = tf.train.Saver()
    SAVER_DIR = "play_model"
    checkpoint_path = os.path.join(SAVER_DIR, "my_model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    
    # initialize
    training_epochs = 5
    batch_size = 30

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            totalbatch = int(data.shape[0]/batch_size)

            for i in range(data.shape[0] - 1):
                c, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
                avg_cost=c/totalbatch
            
            checkpoint_path=saver.save(sess, 'play_model/my_model')

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#        prediction = sess.run(tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: [[3,49,25,96,66,87,30]]})
        print("Learning finished")
        
#screen_record()

#Learning()

def learn2():
    learn_variable=1
    data=np.loadtxt('playdata.csv',delimiter=',', unpack=True, dtype=np.float32)#파일을 읽기전용으로 오픈
    # x_data=data[:-1]
    # y_data=data[-1]

    x_data = np.transpose(data[:3])
    y_data = np.transpose(data[3:])

    print(x_data)
    print(y_data)

    X = tf.placeholder("float", [None, 3])
    Y = tf.placeholder("float", [None, 3])

    # feature별 가중치를 난수로 초기화. feature는 bias 포함해서 3개. 1행 3열.
    W = tf.Variable(tf.zeros([3, 3])) 

    hypothesis = tf.nn.softmax(tf.matmul(X, W))

    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

    learning_rate = 0.01
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    SAVER_DIR = "play_model"
    checkpoint_path = os.path.join(SAVER_DIR, "my_model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    with tf.Session() as sess:
        sess.run(init)

        for step in range(10000):
            sess.run(train, feed_dict={X: x_data, Y: y_data})
            if step % 200 == 0:
                feed = {X: x_data, Y: y_data}
                print('{:4} {:8.6}'.format(step, sess.run(cost, feed_dict=feed)), *sess.run(W))

        checkpoint_path=saver.save(sess, 'play_model/my_model')

    # print('[75] :', sess.run(hypothesis, feed_dict={X: [75]}) > 0.5)
    # print('[175] :', sess.run(hypothesis, feed_dict={X: [175]}) > 0.5)

learn2()

file.close()