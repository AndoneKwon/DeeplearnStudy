import tensorflow as tf
import os
import numpy as np


tf.reset_default_graph()

data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈



def getLearningResult():


    batch_size = 100
    learn_variable=7
    nb_classes=3
    x_data=data[:,0:-1]
    y_data=data[:,[-1]]
    # print(x_data)
    # print(y_data)
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

    # initialize
    training_epochs = 5
    batch_size = 100

    saver = tf.train.Saver()
    SAVER_DIR = "play_model"
    checkpoint_path = os.path.join(SAVER_DIR, "my_model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            prediction = sess.run(tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: [[3,49,25,96,66,87,30]]})
            #print(data)
            print("Prediction: ", prediction)
            # if prediction == 0:        
            #     keyboard.press(Key.up) 
            #     keyboard.release(Key.up)
            # elif prediction == 2:        
            #     keyboard.press_and_release(Key.down)
            sess.close()

getLearningResult()