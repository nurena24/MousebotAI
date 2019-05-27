import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from random import shuffle
import sklearn.model_selection as sk

FILE_I_END = 69

WIDTH = 480
HEIGHT = 270
#LR = 1e-3
LR = 5e-4
EPOCHS = 20

MODEL_NAME = 'MBBigChunks1000v2'
PREV_MODEL = 'MBBigChunks1000v2'

LOAD_MODEL = True

wl = 0
sl = 0
al = 0
dl = 0
nk1 = 0

w = [1, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0]
a = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
nk =[0, 0, 0, 0, 1]

model = googlenet(WIDTH, HEIGHT, 3, LR, output=5, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
    

# iterates through the training files


for e in range(EPOCHS):
    #data_order = [i for i in range(1,FILE_I_END+1)]
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        
        try:
            file_name = 'C:/Users/Nick/PycharmProjects/pygta5/TrainingData/FinalTrainingData/final_training_data-{}.npy'.format(i)
            #file_name = 'C:/Users/Nick/PycharmProjects/pygta5/TrainingData/NewestTrainingData/final_training_data-{}.npy'.format(i)
            # full file info
            train_data = np.load(file_name)
            print('training_data-{}.npy'.format(i),len(train_data))

            size = len(train_data)
            train = train_data[0:int(size*.80)]
            test = train_data[int(size*.80)+1:size]

            X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            test_y = [i[1] for i in test]
            model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)


            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                    
        except Exception as e:
            print(str(e))
            
    








#

#tensorboard --logdir=foo:C:/Users/Nick/PycharmProjects/pygta5/log

