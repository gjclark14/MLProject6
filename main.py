import imblearn as imb
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# https://docs.google.com/document/d/1C8_JHm9Kvd6wW2dxBxbE7rnMKPWwPkMt_8laH9v6Tks/edit?usp=sharing

'''
Authors: 
    Gabriel Clark
    Sassan Estrada
    Naoki Atkins
    
    Machine Learning Project 6 - Classifier Performance on Real-World Data
'''

# 2
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('bank-additional.csv', delimiter=';')
DF = df
print(f'Original dataframe head\n{df.head()}')

# 3
# looking for objects aka categorical variables
# print(f'dataframe types {df.dtypes}')

# target = y, then encode
target = df.y
target = pd.get_dummies(target, columns=['y'], drop_first=True)
# turns target from vector to array, apparently. needed for fit later on
target = np.ravel(target)

# remove columns duration and y before populating dataframe with dummy variables
df = df.drop(columns=['duration', 'y'])

# create a data frame using the object key, anything returning object should be fed to get_dummies
obj_df = df.select_dtypes(include=['object']).copy()
# print(f'object dataframe head\n{obj_df.head()}')

# this should return nominal variables in addition to categorical variables with one-hot encoding
dummies = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                  'day_of_week', 'poutcome'], drop_first=True)
# print(f'dummies head\n{dummies.head()}')

# 4
# print(f'Testing data for upcoming functions.')
# print(f'Dummies Length: {len(dummies)}\nTarget Length: {len(target)}')
# print(f'Dummies contains NaNs? {dummies.isnull().values.any()}')
# print(f'Dummies Datatypes: {dummies.dtypes}')

# Create classifier objects
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))

# Fit each of them
gnb.fit(dummies.to_numpy(), target)
knn.fit(dummies.to_numpy(), target)
svm.fit(dummies.to_numpy(), target)

# Score each of them
print(f'GNB score: {gnb.score(dummies.to_numpy(), target)}')
print(f'KNN score: {knn.score(dummies.to_numpy(), target)}')
print(f'SVM score: {svm.score(dummies.to_numpy(), target)}')

print(f'If closer to 1 means more accurate then KNN has the best score. Otherwise, GNB has the best score.')

# 6
print(f'There are {sum(target)} [yes = 1]\'s in the target set. What does that mean? I have absolutely no clue.')
count = 0
for truth in DF.y:
    if truth != 'no':
        count += 1

print(f'There are {count} yes\'s in the y column of the dataframe.')
