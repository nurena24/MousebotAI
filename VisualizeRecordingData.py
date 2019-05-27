import numpy as np
import os
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import time
#file_name = 'C:/Users/Nick/PycharmProjects/pygta5/TrainingData/balanced_training_data-{}.npy'.format(11)
#train_data = np.load(file_name)
#print(len(train_data))

w = [1, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0]
a = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
nk =[0, 0, 0, 0, 1]
'''dir = 'C:/Users/Nick/PycharmProjects/pygta5/TrainingData/FinalTrainingData/'
files = [x for x in os.listdir(dir) if x.startswith('final_training_data-')]'''
dir = 'C:/Users/Nick/PycharmProjects/pygta5/TrainingData/'
files = [x for x in os.listdir(dir) if x.startswith('training_data-')]
i=0
for file_name in files:
    train_data = np.load(os.path.join(dir,file_name))
    print(file_name)
    print(len(train_data))
    for data in train_data:
        img = data[0]
        print(img.shape)
        choice = data[1]
        cv2.imshow('test',img)
        i=i+1
        if choice == [1, 0, 0, 0, 0]:
            choice_picked = 'straight'
        elif choice == [0, 1, 0, 0, 0]:
            choice_picked = 'reverse'
        elif choice == [0, 0, 1, 0, 0]:
            choice_picked = 'left'
        elif choice == [0, 0, 0, 1, 0]:
            choice_picked = 'right'
        elif choice == [0, 0, 0, 0, 1]:
            choice_picked = 'nokeys'
        print("Frame {} and choice is {}".format(i, choice_picked))
        #print(choice)
        time.sleep(5)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break