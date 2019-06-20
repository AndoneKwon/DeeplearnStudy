import cv2
import time
import numpy as np
import keyboard
import glob
import csv
import tensorflow as tf
import os
#import tensorflow as tf
from pynput.mouse import Controller
from PIL import ImageGrab

file=open("playdata.csv","r",newline='')#파일을 생성 혹은 오픈
csv_write=csv.writer(file)#연 파일을 csv로 바꿔줌

def Learning():
    batch_size = 100
    learn_variable=7
    nb_classes=3
    data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈
    x_data=data[:,0:-1]
    y_data=data[:,[-1]]
    X = tf.placeholder(tf.float32, [None, learn_variable])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, 1])
    
    W1 = tf.Variable(tf.random_normal([learn_variable, 10]))
    b1 = tf.Variable(tf.random_normal([10]))
    layer1=tf.nn.relu(tf.matmul(X,W1)+b1)
    
    W2 = tf.Variable(tf.random_normal([10, 10]))
    b2 = tf.Variable(tf.random_normal([10]))
    layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)


    W3 = tf.Variable(tf.random_normal([10,nb_classes]))
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(layer2, W3) + b3

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

    saver = tf.train.Saver()
    SAVER_DIR = "play_model"
    checkpoint_path = os.path.join(SAVER_DIR, "my_model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    
    # initialize
    training_epochs = 100
    batch_size = 100

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            for i in range(1581):
                c, _ = sess.run([cost, optimizer], feed_dict={
                                X: x_data, Y: y_data})
            checkpoint_path=saver.save(sess, 'play_model/my_model')

            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.9f}'.format(avg_cost))

        print("Learning finished")
    
        
#screen_record()
file.close()
Learning()