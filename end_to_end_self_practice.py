import pandas as pd

housing = pd.read_csv("datasets/housing/housing.csv")
housing.head()

#housing.describe()
housing.info()

housing['ocean_proximity'].value_counts()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
#plt.show()

import numpy as np

np.random.seed(42)
#np.random.permutation(10)
#np.random.seed()

def split_trian_test(data, test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    testsize =  int(len(data)* test_ratio)
    test_indices = shuffle_indices[:testsize]
    train_indices = shuffle_indices[testsize:]
    (train_data, test_data) = (data.iloc[train_indices], data.iloc[test_indices])
    return (train_data, test_data)
    
(train_data, test_data) = split_trian_test(housing, 0.2)    
#print(testsize)
#print(test_indices)
#print(train_indices)
#print(train_data)
#print(test_data)

print(len(train_data))
print(len(test_data))

from sklearn.model_selection import train_test_split

(train_data, test_data) = train_test_split(housing, test_size=0.2, random_state=42)

train_data
#print(len(test_data))

housing['median_income'].hist()