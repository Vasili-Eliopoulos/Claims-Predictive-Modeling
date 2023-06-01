from tkinter import *
from tkinter.filedialog import *

import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import kds

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

filepath = str(askopenfilename())

df = pd.read_excel(filepath)

print(df)

df.shape

print(df.isnull().sum())

for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = df[column].astype(str)
        df[column] = le.fit_transform(df[column])

plt.figure(figsize=(20,20))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})

# plt.show()

corr_matrix = df.corr().abs()
corr = corr_matrix['FraudFound_P']
corr = corr.to_dict()
dropLst = []

for k in corr:
  if corr[k] < 0.00:
    dropLst.append(k)
    print(k + " Removed because of weak correlation")

for column in dropLst:
  df = df.drop(columns=column)

#Bar chart showing distribution of fetal_health classification 
sns.countplot(x = 'FraudFound_P', data = df)

#The distribution is highly unbalanced 
print(df.FraudFound_P.value_counts())

#Randomly oversamples the specified classes to balance the three classes
train, test = train_test_split(df, random_state = 123, test_size = 0.4)

class0 = train[train.FraudFound_P == 0]
class1 = train[train.FraudFound_P == 1]

class0 = resample(class0, replace = True, n_samples = 10000, random_state = 123)
class1 = resample(class1, replace = True, n_samples = 10000, random_state = 123)

train = pd.concat([class0, class1])

print(train.FraudFound_P.value_counts())

x = train.drop('FraudFound_P', axis = 1)
y = train['FraudFound_P']
y = le.fit_transform(y)

x1 = test.drop('FraudFound_P', axis = 1)
y1 = test['FraudFound_P']
y1 = le.fit_transform(y1)

scal = StandardScaler()

x = scal.fit_transform(x)
x1 = scal.transform(x1)

model_accuracy = pd.DataFrame(columns=['Model','Accuracy'])
model = {'XtremeGradientBoost' : XGBClassifier(learning_rate= 0.03, n_estimators= 600, max_depth= 6, subsample= 0.6, colsample_bytree= 1, gamma= 0)}

for test, clf in model.items():
    clf.fit(x,y)
    y_pred = clf.predict(x1)
    acc = accuracy_score(y1, y_pred)
    train_pred = clf.predict(x)
    train_acc = accuracy_score(y, train_pred)
    print(test + ' score' + " {:.1%}".format(round(acc, 3)) + " Prediction Accuracy")
    print('')
    print(classification_report(y1, y_pred))
    cm = confusion_matrix(y1, y_pred, labels = [0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap='Reds',xticks_rotation= 'horizontal' ,colorbar = True)
    plt.grid(False)
    plt.show()
    print('*' * 55,'')
    model_accuracy = pd.concat([model_accuracy, pd.DataFrame({'Model': test, 'Accuracy': round(acc, 3), 'Train_acc': round(train_acc, 3)}, index=[0])], ignore_index=True)

