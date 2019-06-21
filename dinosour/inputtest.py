import tensorflow as tf
import os
import numpy as np


tf.reset_default_graph()

data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈



def getLearningResult():

    learn_variable=7
    nb_classes=2
    data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈
    x_data=data[:,0:-1]
    y_data=data[:, learn_variable:]
    X = tf.placeholder(tf.float32, [None, learn_variable])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, 1])

    dropout_rate=tf.placeholder("float")
    W1 = tf.Variable(tf.random_normal([learn_variable, 20]))
    b1 = tf.Variable(tf.random_normal([20]))
    layer1=tf.nn.relu(tf.matmul(X,W1)+b1)

    W2 = tf.Variable(tf.random_normal([20, 20]))
    b2 = tf.Variable(tf.random_normal([20]))
    layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)
    # layer2=(_layer2,dropout_rate)

    # W3 = tf.Variable(tf.random_normal([20, 20]))
    # b3 = tf.Variable(tf.random_normal([20]))
    # layer3=tf.nn.relu(tf.matmul(layer2,W3)+b3)

    W3 = tf.Variable(tf.random_normal([20, 20]))
    b3 = tf.Variable(tf.random_normal([20]))
    layer3=tf.nn.relu(tf.matmul(layer2,W3)+b3)

    W4 = tf.Variable(tf.random_normal([20,nb_classes]))
    b4 = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(layer3, W4) + b4

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
#            prediction = sess.run(tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: [[6,23,31,96,77,87,36]]})
            prediction = sess.run(tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: [[6,23,31,96,155,87,36]]})
            #print(data)
            print("Prediction: ", prediction)
            # if prediction == 0:        
            #     keyboard.press(Key.up) 
            #     keyboard.release(Key.up)
            # elif prediction == 2:        
            #     keyboard.press_and_release(Key.down)
            sess.close()

getLearningResult()