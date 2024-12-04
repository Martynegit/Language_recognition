import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

IT_x_raw = pd.read_csv("raw_data/IT.txt", sep="\t")
NONIT_x_raw  = pd.read_csv("raw_data/NONIT.txt", sep="\t")
MAXLENGHT = 15


'''TRANSFORM WORDS TO VECTORS'''
def parse_words(words, NORMALIZE=False):
    to_return = np.zeros(( len(words), MAXLENGHT ))
    for i,x in enumerate(words):
        for j,s in enumerate(x):
            to_return[i][j] = ord(s)
    if NORMALIZE==True:
        top = ord('z')
        bot = ord('a') -1
        d = top - bot
        to_return = np.array([[(x - bot)/d if x != 0 else x for x in y] for y in to_return])
    return to_return
    
IT_x = parse_words(IT_x_raw["IT"].values, NORMALIZE=True)
NONIT_x = parse_words(NONIT_x_raw["NONIT"].values, NORMALIZE=True)


'''CREATE SCATTER MATRIX'''
'''
x = pd.DataFrame(IT_x)
scatter_matrix(x)
plt.show()
'''
 
'''ADJUST DATA FOR A ML, CREATE TRAIN/TEST'''
IT_y = np.ones(len(IT_x))
NONIT_y = np.zeros(len(NONIT_x))
IT_l = len(IT_x)
NONIT_l = len(NONIT_x)
percentage = 20 #Percentage of data to use for test set

#TEST SET
#Take the first 20% of IT and NONIT data
IT_top_index = int(IT_l*percentage/100)
NONIT_top_index = int(NONIT_l*percentage/100)
x_test = np.vstack((IT_x[:IT_top_index], NONIT_x[:NONIT_top_index]))
y_test = np.hstack((IT_y[:IT_top_index], NONIT_y[:NONIT_top_index]))

#TRAIN SET
#use only the remaining rows
x_temp = np.vstack((IT_x[IT_top_index:], NONIT_x[NONIT_top_index:]))
y_temp = np.hstack((IT_y[IT_top_index:], NONIT_y[NONIT_top_index:]))

#(non so se serve questa cosa)
#shuffle them and take the first 
indices = np.arange(len(x_temp))
np.random.shuffle(indices) # Shuffle the indices
x = x_temp[indices] #shuffle the arrays
y = y_temp[indices]


'''OUTPUT PARSED DATA FOR FUTURE MODEL TRAINING'''
np.save("parsed_data/x_test", x_test)
np.save("parsed_data/y_test", y_test)
np.save("parsed_data/x_train", x)
np.save("parsed_data/y_train", y)

