import cv2
import sys
import csv
import numpy as np
import keyboard
import glob
import tensorflow as tf
import os
import pyautogui
from PIL import ImageGrab

# tf.reset_default_graph()
# def screen_record():
#     #out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (650,150))

#     # 공룡 사진을 읽음 -> 패턴으로 사용
#     dino = cv2.imread('dino.jpg', 0)
#     w_dino, h_dino = dino.shape[::-1]

#     gameover = cv2.imread('gameover.jpg', 0)

#     # glob 로컬 파일을 읽는 라이브러리 -> 장애물 사진들을 읽음 -> 패턴으로 사용
#     files = glob.glob ('obstacle/*.jpg')
#     obstacles = []
#     for file in files:
#         tmp = cv2.imread(file, 0)
#         obstacles.append(tmp)

#     current_dist = 0
#     pre_dist = 0


#     # 메인 루프
#     while(True):    
#         key_pressed = 'null' # 키값 초기화
#         pre_dist = current_dist # 속도 측정
#         obstacle_index = 0 # 장애물 종류

#         # 스크린 캡쳐 즉 원본화면
#         printscreen = np.array(ImageGrab.grab(bbox=(650,350,1300,500)))
#         #printscreen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))

#         # 현재 화면에 존재하는 모든 장애물을 저장하는 튜플
#         pts = []

#         scr_gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY) # 캡쳐한 이미지를 그레이 톤으로 필터링
#         res_dino = cv2.matchTemplate(scr_gray, dino, cv2.TM_CCOEFF_NORMED) # 필터링 된 이미지에서 공룡 이미지 패턴을 찾고 그 정보를 저장
#         threshold = 0.8 # 유사도의 스레시홀드 그냥 유사도라고 보면 됨 
#         loc_dino = np.where(res_dino >= threshold) # 그 유사도 이상으로 매치된 영역을 찾고 저장

#         for pt in zip(*loc_dino[::-1]):
#             cv2.rectangle(printscreen, pt, (pt[0] + w_dino, pt[1] + h_dino + 9), (50,205,50), 1) # 위에서 찾은 영역을 직사각형 모양으로 원본화면에 표시
#             dinoX = pt[0] + w_dino
#             dinoH = pt[1]

#         #게임오버
#         isGameOver = cv2.matchTemplate(scr_gray, gameover, cv2. TM_CCOEFF_NORMED)
#         w, h = gameover.shape[::-1]
#         loc = np.where(isGameOver >= 0.8)
#         for pt in zip(*loc[::-1]):
#             print('gameover')
#             exit()
          
#         #sys.exit()
          
#         for obstacle in obstacles:
#             res = cv2.matchTemplate(scr_gray, obstacle, cv2.TM_CCOEFF_NORMED) # 장애물도 공룡처럼 찾는다
#             w, h = obstacle.shape[::-1]
#             loc = np.where(res >= 0.8)
#             for pt in zip(*loc[::-1]):
#                 if(pt in pts):
#                     continue
#                 cv2.rectangle(printscreen, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1) # 장애물도 공룡처럼 원본화면에 표시한다
#                 obstacleX, obstacleY = w, h
#                 obstacle_index = obstacles.index(obstacle) # 장애물 종류 저장
#                 pts.append(pt) # 장애물이 있으면 전역변수인 pts에 저장

#         # pts에 저장된 정보 중에서 필요한 정보를 추출
#         if pts:
#             # 키보드 인풋 up(뛰기) down(숙이기) q(종료)
#             if keyboard.is_pressed('up'):
#                 key_pressed = '2'
#             elif keyboard.is_pressed('down'):
#                 key_pressed = '1'
#             elif cv2.waitKey(10) & keyboard.is_pressed('q'): #0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break
#             else:
#                 key_pressed ='0'
#             arr = np.array(pts)
#             current_dist =  min(arr[: ,0]) - dinoX
#             speed = pre_dist - current_dist
#             if speed < 0:
#                 speed = 0
#             #print('obstacle', obstacle_index, 'obstacleX', obstacleX, 'obstacleY', obstacleY, 'obstacleH', min(arr[0, :]), 'dist', current_dist, 'dinoH', dinoH, 'speed', speed, 'pressed', key_pressed)
#         cv2.imshow('window', printscreen)
#         #out.write(printscreen)
#         cv2.waitKey(10)

        
#         data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈
#         learning_rate = 0.001
#         training_epochs = 15
#         batch_size = 100
#         learn_variable=7
#         nb_classes=3
#         x_data=data[:,0:-1]
#         y_data=data[:,[-1]]
#         print(x_data)
#         print(y_data)
#         X = tf.placeholder(tf.float32, [None, learn_variable])
#         # 0 - 9 digits recognition = 10 classes
#         Y = tf.placeholder(tf.float32, [None, 1])

#         W1 = tf.Variable(tf.random_normal([learn_variable, 10]))
#         b1 = tf.Variable(tf.random_normal([10]))
#         layer1=tf.nn.relu(tf.matmul(X,W1)+b1)

#         W2 = tf.Variable(tf.random_normal([10, 10]))
#         b2 = tf.Variable(tf.random_normal([10]))
#         layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)


#         W3 = tf.Variable(tf.random_normal([10,nb_classes]))
#         b3 = tf.Variable(tf.random_normal([nb_classes]))
#         logits = tf.matmul(layer2, W3) + b3

#             # define cost/loss & optimizer
#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

#             # initialize
#         training_epochs = 5
#         batch_size = 100
#         saver = tf.train.Saver()
#         SAVER_DIR = "play_model"
#         checkpoint_path = os.path.join(SAVER_DIR, "my_model")
#         ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
#         with tf.Session() as sess:
#             # Initialize TensorFlow variables
#             sess.run(tf.global_variables_initializer())

#             if ckpt and ckpt.model_checkpoint_path:
#                 saver.restore(sess, ckpt.model_checkpoint_path)    
#                 print("Prediction: ", sess.run(tf.argmax(tf.nn.softmax(logits), 1), feed_dict={X: [[obstacle_index, obstacleX, obstacleY, min(arr[0, :]), current_dist, dinoH, speed]]}))
#                 sess.close()
                   
                        
                    
#             # Training cycle


tf.reset_default_graph()

data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈

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

# learn_variable=1
# nb_classes=2
# data=np.loadtxt('playdata.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈
# x_data=data[:,0:-1]
# y_data=data[:, learn_variable:]
# X = tf.placeholder(tf.float32, [None, learn_variable])
# # 0 - 9 digits recognition = 10 classes
# Y = tf.placeholder(tf.float32, [None, 1])

# # dropout_rate=tf.placeholder("float")
# W1 = tf.Variable(tf.random_normal([learn_variable, nb_classes]))
# b1 = tf.Variable(tf.random_normal([nb_classes]))
# logits=tf.nn.relu(tf.matmul(X,W1)+b1)

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
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=1).minimize(cost)

    # initialize
training_epochs = 200
batch_size = 50

saver = tf.train.Saver()
SAVER_DIR = "play_model"
checkpoint_path = os.path.join(SAVER_DIR, "my_model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

def getLearningResult(input):

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            prediction = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: [input]})
            print(input)
            print("Prediction: ", prediction)
            if prediction == 0:
                pyautogui.keyDown('up')
            elif prediction == 1:
                pyautogui.keyDown('down')
            sess.close()
        
def getPlayData():

    # 공룡 사진을 읽음 -> 패턴으로 사용
    dino = cv2.imread('dino.jpg', 0)
    w_dino, h_dino = dino.shape[::-1]

    gameover = cv2.imread('gameover.jpg', 0)
    is_gameover = False

    # glob 로컬 파일을 읽는 라이브러리 -> 장애물 사진들을 읽음 -> 패턴으로 사용
    files = glob.glob ('obstacle/*.jpg')
    obstacles = []
    for file in files:
        tmp = cv2.imread(file, 0)
        obstacles.append(tmp)

    current_dist = 0
    pre_dist = 0

    # 메인 루프
    while(True):
        pre_dist = current_dist # 속도 측정
        obstacle_index = 0 # 장애물 종류
        obstacleH = 0

        # 스크린 캡쳐 즉 원본화면
        printscreen = np.array(ImageGrab.grab(bbox=(650,350,1300,500)))
        #printscreen = np.array(ImageGrab.grab(bbox=(0,0,1920,1080)))

        # 현재 화면에 존재하는 모든 장애물을 저장하는 튜플
        pts = []

        scr_gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY) # 캡쳐한 이미지를 그레이 톤으로 필터링
        res_dino = cv2.matchTemplate(scr_gray, dino, cv2.TM_CCOEFF_NORMED) # 필터링 된 이미지에서 공룡 이미지 패턴을 찾고 그 정보를 저장
        threshold = 0.8 # 유사도의 스레시홀드 그냥 유사도라고 보면 됨 
        loc_dino = np.where(res_dino >= threshold) # 그 유사도 이상으로 매치된 영역을 찾고 저장

        for pt in zip(*loc_dino[::-1]):
            #cv2.rectangle(printscreen, pt, (pt[0] + w_dino, pt[1] + h_dino + 9), (50,205,50), 1) # 위에서 찾은 영역을 직사각형 모양으로 원본화면에 표시
            dinoX = pt[0] + w_dino
            dinoH = pt[1]

        #게임오버
        GameOver = cv2.matchTemplate(scr_gray, gameover, cv2. TM_CCOEFF_NORMED)
        w, h = gameover.shape[::-1]
        loc = np.where(GameOver >= 0.8)
        for pt in zip(*loc[::-1]):
            print('gameover')
            is_gameover = True
          
        for obstacle in obstacles:
            res = cv2.matchTemplate(scr_gray, obstacle, cv2.TM_CCOEFF_NORMED) # 장애물도 공룡처럼 찾는다
            w, h = obstacle.shape[::-1]
            loc = np.where(res >= 0.8)
            for pt in zip(*loc[::-1]):
                if(pt in pts):
                    continue
                #cv2.rectangle(printscreen, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1) # 장애물도 공룡처럼 원본화면에 표시한다
                obstacleX, obstacleY = w, h
                obstacle_index = obstacles.index(obstacle) # 장애물 종류 저장
                pts.append(pt) # 장애물이 있으면 전역변수인 pts에 저장

        # pts에 저장된 정보 중에서 필요한 정보를 추출
        if pts:
            arr = np.array(pts)
            current_dist =  min(arr[: ,0]) - dinoX
            speed = int((pre_dist - current_dist) / 2)
            obstacleH = min(arr[0, :])
            if current_dist < 0:
                current_dist = 250
            #print('obstacle', obstacle_index, 'obstacleX', obstacleX, 'obstacleY', obstacleY, 'obstacleH', min(arr[0, :]), 'dist', current_dist, 'dinoH', dinoH, 'speed', speed, 'pressed', key_pressed)
            #input = [obstacle_index, obstacleX, obstacleY, min(arr[0, :]), current_dist, dinoH]
        input = [obstacle_index, int(obstacleH / 10), int(current_dist / 30)]
        getLearningResult(input)
        if is_gameover:
            pyautogui.keyDown('space')
            
        #cv2.imshow('window', printscreen)
        cv2.waitKey(0)

getPlayData()