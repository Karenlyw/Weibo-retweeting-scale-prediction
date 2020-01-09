# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:16:15 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:17:41 2019

@author: Administrator
"""

import pandas as pd 
import csv
import numpy as np
from sklearn import svm,metrics,cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.externals import joblib
svm_test_acc=[]
svm_test_rec=[]
svm_test_prec=[]
svm_test_f1=[]
svm_train_acc=[]
svm_train_rec=[]
svm_train_prec=[]
svm_train_f1=[]
file=open(r'C:\Users\Administrator\Desktop\text-cnn-master\text-cnn-master\data\svm_data(score).csv',encoding='gbk')
data = pd.read_csv(file)
x=data[['cnn打分','cos打分','发布者粉丝数','发布者日均发博数','发布者热转率','发布时间段']]
y =data[['传播类别']]
for i in range(0,100):
    x_train,x_test,y_train,y_test=cross_validation.train_test_split(x.values,y.values,test_size=0.2,random_state=i)
    grid = GridSearchCV(svm.SVC(), param_grid={"C":[1, 10,100,1000], "gamma": [0.1, 0.01,0.001,0.0001]}, cv=4)
    grid.fit(x_train,y_train)
    pre_train=grid.predict(x_train)
    pre_test=grid.predict(x_test)
    svm_train_acc.append(metrics.accuracy_score(y_train,pre_train))
    svm_test_acc.append(metrics.accuracy_score(y_test,pre_test))
    lb = preprocessing.LabelBinarizer()   
    y_train=np.array([number[0] for number in lb.fit_transform(y_train)])
    pre_train=np.array([number[0] for number in lb.fit_transform(pre_train)])
    svm_train_rec.append(metrics.recall_score(y_train,pre_train))
    svm_train_prec.append(metrics.precision_score(y_train,pre_train))
    svm_train_f1.append(metrics.f1_score(y_train,pre_train))
    y_test=np.array([number[0] for number in lb.fit_transform(y_test)])
    pre_test=np.array([number[0] for number in lb.fit_transform(pre_test)])
    svm_test_rec.append(metrics.recall_score(y_test,pre_test))
    svm_test_prec.append(metrics.precision_score(y_test,pre_test))
    svm_test_f1.append(metrics.f1_score(y_train,pre_train))
#save model
    #joblib.dump(grid, 'grid.pkl') 
  # print(score2)
   #print(metrics.confusion_matrix(y_test, pre))
  # print(score)

a=0
b=0
c=0
d=0
m=0
n=0
e=0
f=0
for j in svm_train_acc:
    n=n+j
print(n/100)  
for j in svm_train_rec:
    c=c+j
print(c/100)   
for j in svm_train_prec:
    d=d+j
print(d/100)   
for j in svm_train_f1:
    e=e+j
print(e/100)   
for i in svm_test_acc:
    m=m+i
print(m/100)
for i in svm_test_rec:
    a=a+i
print(a/100)
for i in svm_test_prec:
    b=b+i
print(b/100)
for i in svm_test_f1:
    f=f+i
print(f/100)
with open(r'svm.csv','w',newline='') as f:
    file=csv.writer(f)
    file.writerow(['acc','rec','prec','f1'])
    for i in range(0,len(svm_test_acc)):
        file.writerow([svm_test_acc[i],svm_test_rec[i],svm_test_prec[i],svm_test_f1[i]])
    