import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset_filepath_en = '../input/languages-of-europe/en_GB.csv'
dataset_en = pd.read_csv(dataset_filepath_en) #english dataset

dataset_filepath_it = '../input/languages-of-europe/it_IT.csv'
dataset_it = pd.read_csv(dataset_filepath_it) #italian dataset

df_en = pd.DataFrame(dataset_en) 
df_it = pd.DataFrame(dataset_it)

n = 89800
df_it = df_it.iloc[:n] #here I have chopped dataset_it off 

frames = [df_en, df_it]
full_dataset = np.append(frames, []) #"merging" of the datasets
df_data = pd.DataFrame(full_dataset)

en = np.zeros((89800, 1), dtype = int) #array filled with zeros
it = np.ones((89800, 1), dtype = int) #array filled with ones
language = np.concatenate((en, it)) #"merging" of the datasets
lang = pd.DataFrame(language)

df_data["language"] = lang

df_data.columns = ['words/parole', 'language/lingua']
df_data = df_data.sample(frac = 1).reset_index(drop = True) 

df_data.to_csv('data3.csv', index = False, header = False)

X = df_data['words/parole']
y = df_data['language/lingua']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 138729, test_size = 0.20) #training and test set

X_train.to_csv('X_train.csv', index = False, header = False)
X_test.to_csv('X_test.csv', index = False, header = False)
y_train.to_csv('y_train.csv', index = False, header = False)
y_test.to_csv('y_test.csv', index = False, header = False)
