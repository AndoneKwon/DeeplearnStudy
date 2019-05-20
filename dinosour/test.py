import cv2
import time
import numpy as np
import keyboard
import glob
import csv
#import tensorflow as tf
from pynput.mouse import Controller
from PIL import ImageGrab

file=open("C:/Users/권진우/Documents/GitHub/DeeplearnStudy/dinosour/data.csv","a",newline='')#파일을 생성 혹은 오픈
csv_write=csv.writer(file)#연 파일을 csv로 바꿔줌

def screen_record():
    #out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (650,150))

    last_time = time.time()
    # 공룡 사진을 읽음 -> 패턴으로 사용
    dino = cv2.imread('dino.jpg', 0)
    w_dino, h_dino = dino.shape[::-1]#image shaple을 너비와 높이를 저장한다. 값은 3차원 행렬로 return 된다.

    # glob 로컬 파일을 읽는 라이브러리 -> 장애물 사진들을 읽음 -> 패턴으로 사용
    files = glob.glob ('obstacle/*.jpg')
    obstacles = []
    for file in files:
        tmp = cv2.imread(file, 0)
        obstacles.append(tmp)
    print('press up and down')
    while True:
        if keyboard.is_pressed('up'):
            print('up is pressed')
            (x0, y0) = Controller().position
            break

    while True:
        if keyboard.is_pressed('down'):
            print('up is down')
            (x1, y1) = Controller().position
            break
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

    # 메인 루프
    while(True):
        # 스크린 캡쳐 즉 원본화면
        printscreen = np.array(ImageGrab.grab(bbox=(p0,p1,p2,p3)))

        # 현재 화면에 존재하는 모든 장애물을 저장하는 튜플
        pts = []

        # 키보드 인풋 up(뛰기) down(숙이기) q(종료)
#        if keyboard.is_pressed('up'):
#            print('up')
#        elif keyboard.is_pressed('down'):
#            print('down')
#        elif cv2.waitKey(20) & keyboard.is_pressed('q'): #0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break

        scr_gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY) # 캡쳐한 이미지를 그레이 톤으로 필터링
        res_dino = cv2.matchTemplate(scr_gray, dino, cv2.TM_CCOEFF_NORMED) # 필터링 된 이미지에서 공룡 이미지 패턴을 찾고 그 정보를 저장
        threshold = 0.8 # 유사도의 스레시홀드 그냥 유사도라고 보면 됨 
        loc_dino = np.where(res_dino >= threshold) # 그 유사도 이상으로 매치된 영역을 찾고 저장

        for pt in zip(*loc_dino[::-1]):#zip 안에 있는 정보 대입 하며 반복
            cv2.rectangle(printscreen, pt, (pt[0] + w_dino, pt[1] + h_dino + 9), (50,205,50), 1) # 위에서 찾은 영역을 직사각형 모양으로 원본화면에 표시
            dinoX = pt[0] + w_dino
            dinoH = pt[1]

        for obstacle in obstacles:
            res = cv2.matchTemplate(scr_gray, obstacle, cv2.TM_CCOEFF_NORMED) # 장애물도 공룡처럼 찾는다
            w, h = obstacle.shape[::-1]
            loc = np.where(res >= 0.8)
            for pt in zip(*loc[::-1]):
                if(pt in pts):
                    continue
                cv2.rectangle(printscreen, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1) # 장애물도 공룡처럼 원본화면에 표시한다
                pts.append(pt) # 장애물이 있으면 전역변수인 pts에 저장

        # pts에 저장된 정보 중에서 필요한 정보를 추출
        if pts:
            arr = np.array(pts)
            dist=min(arr[: ,0])-dinoX
            height=min(arr[0, :])
            print('dinoH', dinoH, 'dist', dist, ' height', height)
            csv_write.writerow([dinoH,dist,height])
            if keyboard.is_pressed('up'):
                print('up')
            elif keyboard.is_pressed('down'):
                print('down')
            elif cv2.waitKey(20) & keyboard.is_pressed('q'): #0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        cv2.imshow('window', printscreen)
        #out.write(printscreen)
        cv2.waitKey(20)

        last_time = time.time()
        

def Learning():
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    learn_variable=7
    nb_classes=3
    data=np.loadtxt('C:/Users/권진우/Documents/GitHub/DeeplearnStudy/dinosour/data.csv',delimiter=',',dtype=np.float32)#파일을 읽기전용으로 오픈
    x_data=data[:,0:-1]
    y_data=data[:,[-1]]
    X = tf.placeholder(tf.float32, [None, learn_variable])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, 3])
    
    W1 = tf.Variable(tf.random_normal([learn_variable, 10]))
    b1 = tf.Variable(tf.random_normal([10]))
    layer1=tf.nn.relu(tf.matmul(X,W1)+b1)
    
    W2 = tf.Variable(tf.random_normal([10, 10]))
    b2 = tf.Variable(tf.random_normal([10]))
    layer2=tf.nn.relu(tf.matmul(layer1,W2)+b2)


    W3 = tf.Variable(tf.random_normal([10,nb_classes]))
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)

    # define cost/loss & optimizer
    cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: x_data, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
    
        
#screen_record()
file.close()
print(x_data.shape,x_data)
print(y_data.shape,y_data)