import cv2
import sys
import csv
import numpy as np
import keyboard
import glob
import time
from PIL import ImageGrab


#데이터 저장
csvfile = open('playdata.csv', 'a', newline='') # 이어쓰기모드
#csvfile = open('playdata.csv', 'w', newline='') # 새로쓰기모드
csvwriter = csv.writer(csvfile)

def screen_record():
    #out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (650,150))



    # 공룡 사진을 읽음 -> 패턴으로 사용
    dino = cv2.imread('dino.jpg', 0)
    w_dino, h_dino = dino.shape[::-1]

    gameover = cv2.imread('gameover.jpg', 0)

    # glob 로컬 파일을 읽는 라이브러리 -> 장애물 사진들을 읽음 -> 패턴으로 사용
    files = glob.glob ('obstacle/*.jpg')
    obstacles = []
    for file in files:
        tmp = cv2.imread(file, 0)
        obstacles.append(tmp)

    current_dist = 0
    pre_dist = 0


    start = time.time()

    # 메인 루프
    while(True):
        up_pressed = 0 # 키값 초기화
        down_pressed = 0
        not_pressed = 0
        key_pressed = 0
        pre_dist = current_dist # 속도 측정
        obstacle_index = 0 # 장애물 종류

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
            cv2.rectangle(printscreen, pt, (pt[0] + w_dino, pt[1] + h_dino + 9), (50,205,50), 1) # 위에서 찾은 영역을 직사각형 모양으로 원본화면에 표시
            dinoX = pt[0] + w_dino
            dinoH = pt[1]

        #게임오버
        isGameOver = cv2.matchTemplate(scr_gray, gameover, cv2. TM_CCOEFF_NORMED)
        w, h = gameover.shape[::-1]
        loc = np.where(isGameOver >= 0.8)
        for pt in zip(*loc[::-1]):
            print('gameover')
            csvfile.close()
            exit()
          
        #sys.exit()
          
        for obstacle in obstacles:
            res = cv2.matchTemplate(scr_gray, obstacle, cv2.TM_CCOEFF_NORMED) # 장애물도 공룡처럼 찾는다
            w, h = obstacle.shape[::-1]
            loc = np.where(res >= 0.8)
            for pt in zip(*loc[::-1]):
                if(pt in pts):
                    continue
                cv2.rectangle(printscreen, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1) # 장애물도 공룡처럼 원본화면에 표시한다
                obstacleX, obstacleY = w, h
                obstacle_index = obstacles.index(obstacle) # 장애물 종류 저장
                pts.append(pt) # 장애물이 있으면 전역변수인 pts에 저장

        # pts에 저장된 정보 중에서 필요한 정보를 추출
        if pts:
            # 키보드 인풋 up(뛰기) down(숙이기) q(종료)
            if keyboard.is_pressed('up'):
                up_pressed = 1
            elif keyboard.is_pressed('down'):
                down_pressed = 1
            elif cv2.waitKey(10) & keyboard.is_pressed('q'): #0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
                not_pressed = 1
            arr = np.array(pts)
            current_dist =  min(arr[: ,0]) - dinoX
            if current_dist < 0:
                current_dist = 250
            speed = pre_dist - current_dist
            if speed < 0:
                speed = 0
            #print('obstacle', obstacle_index, 'obstacleX', obstacleX, 'obstacleY', obstacleY, 'obstacleH', min(arr[0, :]), 'dist', current_dist, 'dinoH', dinoH, 'speed', speed, 'pressed', key_pressed)
            #csvwriter.writerow([obstacle_index, obstacleX, obstacleY, min(arr[0, :]), current_dist, dinoH, key_pressed])
            csvwriter.writerow([obstacle_index, int(min(arr[0, :]) / 10), int(current_dist / 30), up_pressed, down_pressed, not_pressed])
        cv2.imshow('window', printscreen)
        #out.write(printscreen)
        cv2.waitKey(10)

screen_record()

csvfile.close()