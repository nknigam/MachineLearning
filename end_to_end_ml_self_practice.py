# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

housing = pd.read_csv("datasets/housing/housing.csv")
housing.head()


# %%
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


# %%
(train_data, test_data) = split_trian_test(housing, 0.2)   


# %%
from sklearn.model_selection import train_test_split

(train_data, test_data) = train_test_split(housing, test_size=0.2, random_state=42)
#train_data
#print(len(test_data))


# %%
housing['median_income'].hist()


# %%
housing['income_category'] = pd.cut(housing['median_income'], bins = [0,1.5,3.0,4.5,6.0,np.inf], labels = ['cat1', 'cat2', 'cat3', 'cat4','cat5'])


# %%
print(housing['income_category'].value_counts())


# %%
housing['income_category'].hist()


# %%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

strata = split.split(housing, housing["income_category"])
for train_indices, test_indices in strata:
    strata_train_set = housing.loc[train_indices]
    strata_test_set = housing.loc[test_indices]


# %%
#strata_train_set['income_category'].value_counts()
strata_test_set['income_category'].value_counts() / len(strata_test_set)


# %%
housing["income_category"].value_counts() / len(housing)


# %%
def income_cat_proportions(data):
    return data["income_category"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strata_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100


# %%
compare_props


# %%
strata_train_set.head()


# %%


