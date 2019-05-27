import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import random

random.seed()
FILE_I_END = 113

def balance_data(data_in):
    # Assume data is not malformed and num choices is consistent
    num_choices = len(data_in[0][1])

    # Split each choice into seperate lists
    choice_lists = []
    for i in range(num_choices):
        choice_lists.append([c for c in data_in if c[1][i]])

    # Find the choice least represented to balance the data to
    min_len = [len(l) for l in choice_lists]
    min_len=(np.trim_zeros(min_len))#trim zeros from array. ex[122,102,80,0,0]
    min_len.remove(min(min_len))#remove the lowest number from array(jumps)[122,102,80]
    min_len = min(min_len)  # Get the minimum of the array[80]
    print("minimum lenght is=",min_len)
    # Trim all choices to new length and shuffle resulting data
    data_out = []
    for l in choice_lists:
        data_out += l[:min_len]
    shuffle(data_out)

    return data_out


data_order = [i for i in range(1, FILE_I_END + 1)]
shuffle(data_order)
for count, i in enumerate(data_order):
    try:
        random.seed()
        file_name = 'training_data-{}.npy'.format(i)
        # full file info
        train_data = np.load(file_name, allow_pickle=True)
        print('training_data-{}.npy'.format(i), len(train_data))
        df = pd.DataFrame(train_data)
        #print(df.head())
        print(Counter(df[1].apply(str)))
        final_data = balance_data(train_data)
        np.save('balanced_training_data-{}.npy'.format(i), final_data)

    except Exception as e:
        print(str(e))