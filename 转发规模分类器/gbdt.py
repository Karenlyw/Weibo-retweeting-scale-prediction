# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:44:54 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import csv
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.externals import joblib

gbdt_test_acc=[]
gbdt_test_rec=[]
gbdt_test_prec=[]
gbdt_test_f1=[]
gbdt_train_acc=[]
gbdt_train_rec=[]
gbdt_train_prec=[]
gbdt_train_f1=[]
file= open(r'C:\Users\Administrator\Desktop\text-cnn-master\text-cnn-master\data\gbdt_data(score).csv',encoding='gbk')
data = pd.read_csv(file)
x=data[['cnn打分','cos打分','发布者粉丝数','发布者日均发博数','发布者热转率','发布时间段']]
y =data[['传播类别']]
time_score=0
video_score=0
cnn_score=0
fans_score=0
daily_score=0
popular_score=0

for i in range(0,100):
    x_train,x_test,y_train,y_test=cross_validation.train_test_split(x.values,y.values,test_size=0.2,random_state=i)

    
                      
    model= GradientBoostingClassifier()
    model.fit(x_train,y_train)
    
    time_score=time_score+(model.feature_importances_)[0]
    video_score=video_score+(model.feature_importances_)[1]
    cnn_score=cnn_score+(model.feature_importances_)[2]
    fans_score=fans_score+(model.feature_importances_)[3]
    daily_score=daily_score+(model.feature_importances_)[4]
    popular_score=popular_score+(model.feature_importances_)[5]
    
#grid = joblib.load('grid3.pkl')
    pre_train=model.predict(x_train)
    pre_test=model.predict(x_test)
    gbdt_train_acc.append(metrics.accuracy_score(y_train,pre_train))
    gbdt_test_acc.append(metrics.accuracy_score(y_test,pre_test))
    lb = preprocessing.LabelBinarizer()
    y_train=np.array([number[0] for number in lb.fit_transform(y_train)])
    pre_train=np.array([number[0] for number in lb.fit_transform(pre_train)])

    gbdt_train_rec.append(metrics.recall_score(y_train,pre_train))
    gbdt_train_prec.append(metrics.precision_score(y_train,pre_train))
    gbdt_train_f1.append(metrics.f1_score(y_train,pre_train))
    y_test=np.array([number[0] for number in lb.fit_transform(y_test)])
    pre_test=np.array([number[0] for number in lb.fit_transform(pre_test)])
    gbdt_test_rec.append(metrics.recall_score(y_test,pre_test))
    gbdt_test_prec.append(metrics.precision_score(y_test,pre_test))
    gbdt_test_f1.append(metrics.f1_score(y_train,pre_train))
#save model
    if(i==52):
        joblib.dump(model,'retweeting prediction.pkl') 
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
for j in gbdt_train_acc:
    n=n+j
print(n/100)  
for j in gbdt_train_rec:
    c=c+j
print(c/100)   
for j in gbdt_train_prec:
    d=d+j
print(d/100)   
for j in gbdt_train_f1:
    e=e+j
print(e/100)   
for i in gbdt_test_acc:
    m=m+i
print(m/100)
for i in gbdt_test_rec:
    a=a+i
print(a/100)
for i in gbdt_test_prec:
    b=b+i
print(b/100)
for i in gbdt_test_f1:
    f=f+i
print(f/100)
print("score of time,video,cnn,fans,daily,popular")
print(time_score/100)
print(video_score/100)
print(cnn_score/100)
print(fans_score/100)
print(daily_score/100)
print(popular_score/100)