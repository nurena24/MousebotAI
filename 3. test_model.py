import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from models import inception_v3 as googlenet
from getkeys import key_check
from collections import deque, Counter
import random
from statistics import mode, mean
import numpy as np
from motion import motion_detection

GAME_WIDTH = 1920
GAME_HEIGHT = 1080

how_far_remove = 800
rs = (20, 15)
log_len = 25

motion_req = 800
motion_log = deque(maxlen=log_len)

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10

choices = deque([], maxlen=5)
hl_hist = 250
choice_hist = deque([], maxlen=hl_hist)

w = [1, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0]
a = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
nk =[0, 0, 0, 0, 1]


t_time = 0.25


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():

    ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    # ReleaseKey(S)


def right():
    ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)




def no_keys():

    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


model = googlenet(WIDTH, HEIGHT, 3, LR, output=5)
MODEL_NAME = 'MBBigChunks1000v2'
model.load(MODEL_NAME)

print('We have loaded a previous model!!!!')


def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT + 40))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen, (WIDTH, HEIGHT))

    t_minus = prev
    t_now = prev
    t_plus = prev

    while (True):

        if not paused:
            screen = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT + 40))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 3)])[0]
            prediction = np.array(prediction) * np.array([1, 1, 1,1, 0.85])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'

            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'

            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                no_keys()
                choice_picked = 'nokeys'


            print('loop took {} seconds. Choice: {}'.format(round(time.time() - last_time, 3),
                                                                        choice_picked))


        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


main()