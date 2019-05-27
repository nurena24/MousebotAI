#This takes all the balanced data files and mergest them into 1 giant file. Had to do this because the
#balanced files were of small sizes..too small to feed to model.fit. Then I split up the giant file evenly into 500 row smaller files
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import random

random.seed()
FILE_I_END = 113

big_list=[]
running_count=0
data_order = [i for i in range(1, FILE_I_END + 1)]
shuffle(data_order)
for count, i in enumerate(data_order):
    try:
        random.seed()
        file_name = 'C:/Users/Nick/PycharmProjects/pygta5/TrainingData/balanced_training_data-{}.npy'.format(i)
        train_data = np.load(file_name, allow_pickle=True)
        big_list.extend(train_data)
        running_count = running_count+len(train_data)
        print(file_name)
        print("size of BigList=",running_count)
    except Exception as e:
        print(str(e))
print("Final BigList=",len(big_list))
#np.save('_BigList_balanced_training_data-{}.npy'.format(1), big_list)


print("now chunks")
chunkSize = 1000
chunks = [big_list[x:x+chunkSize] for x in range(0, len(big_list), chunkSize)]

starting_value = 70
for idx, chunk in enumerate(chunks):
    print(idx)
    print(len(chunk))
    np.save('final_training_data-{}.npy'.format(idx+starting_value), chunk)
  #df = pd.DataFrame(chunk)
  #print(Counter(df[1].apply(str)))
  #print(len(chunk))

