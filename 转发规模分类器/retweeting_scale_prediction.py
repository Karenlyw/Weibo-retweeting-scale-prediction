# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:09:24 2019

@author: Administrator
"""
from sklearn.externals import joblib
print("请输入cnn打分")
a=input()
print("请输入cos打分")
b=input()
print("请输入发布者粉丝数")
c=input()
print("请输入发布者日均发博数")
d=input()
print("请输入发布者热转率")
e=input()
print("发布时间段")
f=input()
x=[[a,b,c,d,e,f]]
model = joblib.load('retweeting prediction.pkl')
pre_train=model.predict(x)
print("预测转发规模为：")
print(pre_train[0])