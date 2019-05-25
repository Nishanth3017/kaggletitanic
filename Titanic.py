#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#datasets
dataset = pd.read_csv('train.csv')
X_target = dataset.iloc[:,[1]].values
X_required = dataset.iloc[:, [2,4,5,9]].values

#missingdata
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X_required[:, [2]])
X_required[:, [2]] = imputer.transform(X_required[:, [2]])

#seperate the data
X=X_required
y=X_target

#categoricaldata
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

#training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#linear svc
from sklearn import svm
clf = svm.LinearSVC()

#train our model
clf.fit(X_train,y_train)

#predict
clf.predict(X_test[0:10])

#accuracy
clf.score(X_test,y_test)